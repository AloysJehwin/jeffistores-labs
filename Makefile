.PHONY: help setup gpu-check lint format jupyter clean

help:
	@echo "Targets:"
	@echo "  setup       Install env via uv (CPU-friendly base)"
	@echo "  setup-hf    Add HuggingFace stack (transformers, peft, trl)"
	@echo "  gpu-check   Print CUDA / MPS availability"
	@echo "  jupyter     Launch Jupyter Lab"
	@echo "  lint        Ruff lint"
	@echo "  format      Ruff format"
	@echo "  clean       Remove caches and build artifacts"

setup:
	uv sync

setup-hf:
	uv sync --extra hf --extra track

gpu-check:
	uv run python -c "import torch; \
	cuda=torch.cuda.is_available(); \
	mps=getattr(torch.backends,'mps',None) and torch.backends.mps.is_available(); \
	print(f'PyTorch: {torch.__version__}'); \
	print(f'CUDA:    {cuda}{\" | \"+torch.cuda.get_device_name(0) if cuda else \"\"}'); \
	print(f'MPS:     {bool(mps)}')"

jupyter:
	uv run jupyter lab

lint:
	uv run ruff check .

format:
	uv run ruff format .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__ build dist *.egg-info
