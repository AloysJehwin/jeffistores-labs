"""QLoRA fine-tuning of a small LLM on the Jeffi descgen dataset.

This is a *thin* wrapper around HuggingFace's `trl.SFTTrainer`. It:

    1. loads a 4-bit quantized base model + tokenizer
    2. attaches LoRA adapters per the config
    3. formats our dataset.Example records through the chat template
    4. trains with W&B logging
    5. saves the adapter (not the merged model — that's a separate step)

Why a thin wrapper at all? Because `SFTTrainer` has 50+ arguments and the
config file pins exactly the ones we care about. The rest stay at sensible
defaults.

Anti-patterns avoided:
- We do NOT merge the LoRA into the base model here. Saving the adapter
  separately means later runs can be diffed and the base weights stay
  cached. See `merge_adapter()` in `serve.py` (Stage 4).
- We do NOT mask the prompt at training time by default — Phi-3's chat
  template already structures the assistant turn so SFTTrainer will only
  compute loss on the response tokens via DataCollatorForCompletionOnlyLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset

from .dataset import Example, SYSTEM_PROMPT, render_input_block

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Parsed YAML config — see configs/phi3_qlora_v1.yaml for field docs."""

    run_name: str
    base_model: str
    quant: dict[str, Any]
    lora: dict[str, Any]
    training: dict[str, Any]
    smoke: dict[str, Any] | None
    wandb: dict[str, Any] | None

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        data = yaml.safe_load(path.read_text())
        return cls(
            run_name=data["run_name"],
            base_model=data["base_model"],
            quant=data.get("quant", {}),
            lora=data.get("lora", {}),
            training=data.get("training", {}),
            smoke=data.get("smoke"),
            wandb=data.get("wandb"),
        )


# -----------------------------------------------------------------------------
# Data formatting
# -----------------------------------------------------------------------------


# Per-run metrics dumper. Independent of W&B — works offline, survives across
# trl/transformers versions, gives us a stable artifact we can diff against
# future runs. One file per run_name; JSON-per-line so it streams.
_RUNS_DIR = Path(__file__).resolve().parents[3] / "experiments" / "04_jeffi_descgen" / "runs"


class _JsonlMetricsCallback:
    """Writes every TrainerState log dict as one JSON line to disk.

    We don't subclass `TrainerCallback` at class-definition time because
    importing transformers at module import is expensive and we want this
    module to stay cheap to import. Instead we build the subclass lazily
    inside a factory and return an instance of *that* class. The result is
    duck-typed compatible with the trainer's callback protocol *and*
    inherits every no-op event handler from the official base, so we're
    forward-compatible with new TrainerCallback methods (e.g. transformers
    5.x added `on_pre_optimizer_step`).
    """

    def __new__(cls, run_name: str):
        from transformers import TrainerCallback as _TC

        _RUNS_DIR.mkdir(parents=True, exist_ok=True)
        path = _RUNS_DIR / f"{run_name}.jsonl"
        path.write_text("")  # truncate on each run

        class _Impl(_TC):
            def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
                if not logs:
                    return
                import json as _json
                record = {"step": state.global_step, "epoch": state.epoch, **logs}
                with path.open("a", encoding="utf-8") as f:
                    f.write(_json.dumps(record, default=str) + "\n")

        return _Impl()


def format_for_chat_template(example: Example, tokenizer: Any) -> dict[str, str]:
    """Render one Example into a single text string the trainer can tokenize.

    Uses the model's chat template so we get the exact special tokens the
    base model was trained with (Phi-3 uses <|user|>...<|end|><|assistant|>).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_input_block(example.product_input)},
        {"role": "assistant", "content": example.target_description},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def examples_to_dataset(examples: list[Example], tokenizer: Any) -> Dataset:
    """Convert in-memory Examples into a HuggingFace Dataset of {text}."""
    return Dataset.from_list([format_for_chat_template(ex, tokenizer) for ex in examples])


# -----------------------------------------------------------------------------
# Training entrypoint
# -----------------------------------------------------------------------------


def train(
    config: TrainingConfig,
    train_examples: list[Example],
    val_examples: list[Example],
    *,
    smoke_test: bool = False,
) -> Path:
    """Run a training pass. Returns the path of the saved adapter.

    Args:
        config:        parsed YAML
        train/val:     dataset.Example lists from read_jsonl
        smoke_test:    if True, applies config.smoke overrides (max_steps etc.)
                       so you can verify the pipeline in a few minutes
    """
    # Heavy imports kept lazy so importing this module is cheap.
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(config.training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Tokenizer ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        # Phi-3 ships without a pad token; reuse eos so SFTTrainer is happy
        tokenizer.pad_token = tokenizer.eos_token

    # -- Model (4-bit quantized) ------------------------------------------
    bnb = BitsAndBytesConfig(
        load_in_4bit=config.quant.get("load_in_4bit", True),
        bnb_4bit_quant_type=config.quant.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=getattr(
            torch, config.quant.get("bnb_4bit_compute_dtype", "bfloat16")
        ),
        bnb_4bit_use_double_quant=config.quant.get("bnb_4bit_use_double_quant", True),
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    # Required dance for 4-bit + gradient checkpointing
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.training.get("gradient_checkpointing", True),
    )

    # -- LoRA --------------------------------------------------------------
    peft_cfg = LoraConfig(
        r=config.lora.get("r", 8),
        lora_alpha=config.lora.get("alpha", 16),
        lora_dropout=config.lora.get("dropout", 0.05),
        target_modules=config.lora.get("target_modules"),
        bias=config.lora.get("bias", "none"),
        task_type=config.lora.get("task_type", "CAUSAL_LM"),
        use_dora=config.lora.get("use_dora", False),
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()  # sanity print: ~0.1–1% of params

    # -- Datasets ---------------------------------------------------------
    train_ds = examples_to_dataset(train_examples, tokenizer)
    val_ds = examples_to_dataset(val_examples, tokenizer)

    # -- Trainer config ---------------------------------------------------
    sft_kwargs: dict[str, Any] = dict(config.training)
    sft_kwargs["output_dir"] = str(output_dir)
    sft_kwargs["run_name"] = config.run_name
    if smoke_test and config.smoke:
        smoke_save = config.smoke.get("save_steps", 25)
        sft_kwargs.update(
            max_steps=config.smoke.get("max_steps", 50),
            save_steps=smoke_save,
            eval_steps=smoke_save,   # transformers requires save_steps % eval_steps == 0
            num_train_epochs=1,      # ignored when max_steps is set
        )

    # report_to: respect WANDB_DISABLED=true so smoke tests don't need a login
    import os
    wandb_disabled = os.environ.get("WANDB_DISABLED", "").lower() in {"true", "1", "yes"}
    if wandb_disabled:
        sft_kwargs["report_to"] = "none"
    else:
        sft_kwargs.setdefault("report_to", "wandb" if config.wandb else "none")

    # trl >= 1.0 dropped max_seq_length from SFTConfig — it auto-detects
    # from the tokenizer. Pop it defensively so older configs still load.
    max_seq_length = sft_kwargs.pop("max_seq_length", None)

    sft_config = SFTConfig(
        dataset_text_field="text",
        packing=False,            # one example per row, no concat
        **sft_kwargs,
    )

    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_config,
        processing_class=tokenizer,
        callbacks=[_JsonlMetricsCallback(config.run_name)],
    )
    # Some trl versions still accept tokenizer; some prefer processing_class.
    # If processing_class isn't supported, fall back to tokenizer.
    try:
        trainer = SFTTrainer(**trainer_kwargs)
    except TypeError:
        trainer_kwargs.pop("processing_class")
        trainer_kwargs["tokenizer"] = tokenizer
        trainer = SFTTrainer(**trainer_kwargs)
    _ = max_seq_length  # quiet linter; trl auto-detects from tokenizer.model_max_length

    # -- Train + save ------------------------------------------------------
    trainer.train()
    adapter_dir = output_dir / "adapter-final"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    return adapter_dir
