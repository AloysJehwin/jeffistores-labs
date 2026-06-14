# data/

Local datasets. **Gitignored** — keep these on Ubuntu (the GPU box).

Structure (created as needed):
```
data/
  tinyshakespeare/      For nanoGPT
  jeffi_descriptions/   Exported product descriptions for fine-tuning
  hf_cache/             HuggingFace model + dataset cache (set HF_HOME here)
```

To export Jeffi product data on Ubuntu (later, when needed):
```bash
# On the EC2 box or via SSH tunnel — see scripts/export_jeffi_descriptions.py
```
