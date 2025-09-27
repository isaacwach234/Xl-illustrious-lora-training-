# Illustrious XL Releases

This reference lists every Illustrious XL base release supported by the notebook (v0.1 onward), the primary Hugging Face repository for each, and practical notes gathered from community usage reports and guides.【7e84be†L1-L38】【e47aa3†L15-L63】

| Release | Hugging Face repo ID | Primary checkpoint file | SHA256 | Key notes |
| --- | --- | --- | --- | --- |
| v0.1 | `OnomaAIResearch/Illustrious-xl-early-release-v0` | `Illustrious-XL-v0.1.safetensors` | — | Base LoRA training target; SDPA recommended during training.【b66993†L1-L16】【e4dc23†L23-L39】 |
| v1.0 | `Liberata/illustrious-xl-v1.0` | `Illustrious-XL-v1.0.safetensors` | — | Higher-resolution (up to 1024×1536) prompting, still tag-first; mixes remain compatible with v0.1 LoRAs.【7b6859†L13-L64】【e4aa15†L1-L18】【e47aa3†L15-L44】 |
| v1.1 | `OnomaAIResearch/Illustrious-XL-v1.1` | `Illustrious-XL-v1.1.safetensors` | — | Adds partial natural-language prompting alongside tags; continue to prefer tag scaffolding.【3674e3†L1-L22】【e47aa3†L15-L44】 |
| v2.0 | `OnomaAIResearch/Illustrious-XL-v2.0` | `Illustrious-XL-v2.0.safetensors` | `1eaab5894d2f38977d6319b175fdf2fec4911287160c2eaaaceb09870c1f66f0` | Best with ancestral/CFG++ samplers; improved high-res coherence; packaged with SDXL base/offset files.【015d51†L11-L17】【f2f746†L1-L40】【e47aa3†L44-L61】

## Release details & prerequisites

### v0.1 – early release
- Repo & file: `OnomaAIResearch/Illustrious-xl-early-release-v0` → `Illustrious-XL-v0.1.safetensors` (direct Hugging Face link used in training workflows).【b66993†L1-L16】
- Training utilities enable SDPA explicitly for stability; keep SDPA or xformers active when fine-tuning.【e4dc23†L23-L39】
- Remains the canonical base for backwards-compatible LoRA training despite newer checkpoints.【7b6859†L45-L64】

### v1.0 – Liberata fork
- Repo & file: `Liberata/illustrious-xl-v1.0` hosting `Illustrious-XL-v1.0.safetensors`; community notebooks add this repo alongside v0.1 for better prompt adherence.【7b6859†L13-L64】
- Training logs show the checkpoint loads cleanly inside SDXL pipelines (xFormers enabled by default).【e4aa15†L1-L18】
- According to community guidance, v1.0 expands reliable resolutions to 1024×1536 and beyond while keeping tag-first prompting best practice.【e47aa3†L15-L44】

### v1.1 – OnomaAIResearch refresh
- Repo & file: `OnomaAIResearch/Illustrious-XL-v1.1` → `Illustrious-XL-v1.1.safetensors`; used directly in downstream SDKs such as DiffSynth Studio.【3674e3†L1-L18】
- Maintains v1.0’s higher resolution targets and natural-language assistance, but tags remain essential for consistent control.【e47aa3†L15-L44】

### v2.0 – current flagship
- Repo & file: `OnomaAIResearch/Illustrious-XL-v2.0` → `Illustrious-XL-v2.0.safetensors`; accessible via direct huggingface download links in automation tooling.【015d51†L11-L17】
- Published metadata exposes the checkpoint hash `1eaab5894d2f38977d6319b175fdf2fec4911287160c2eaaaceb09870c1f66f0`, plus bundled SDXL base/offset assets for diffusers pipelines.【f2f746†L1-L40】
- Guides recommend samplers with noise injection (e.g., Euler A, res_multistep_ancestral_cfgpp ≈1.5–1.7 CFG) and highlight improved high-res prompt adherence compared with earlier releases.【e47aa3†L44-L61】

## General setup notes
- Large Hugging Face downloads routinely authenticate with a user token (see SD.Next logs showing token use during Illustrious pipeline setup); set the `HUGGING_FACE_HUB_TOKEN` env var or login via `huggingface-cli login` before running the notebook.【a87993†L1-L4】
- Keep SDPA/flash attention enabled when possible—community training logs explicitly enable SDPA on the UNet for Illustrious XL checkpoints to prevent instability.【e4dc23†L23-L39】

## Example: fetching a specific release
```python
from huggingface_hub import snapshot_download

# Replace with the release you need, e.g. Liberata/illustrious-xl-v1.0
snapshot_download(
    repo_id="OnomaAIResearch/Illustrious-XL-v2.0",
    allow_patterns=["Illustrious-XL-v2.0.safetensors"],
    local_dir="./models/illustrious-xl-v2.0",
    token=os.environ.get("HUGGING_FACE_HUB_TOKEN")  # ensure you are authenticated
)
```
The `allow_patterns` filter keeps downloads focused on the checkpoint file if you already have accompanying VAE or config assets; omit it to grab the full repo (including bundled SDXL base files documented in the v2.0 model card).【f2f746†L1-L40】
