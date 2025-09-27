# Error handling guide

This guide summarizes the failure cases that **Illustriousxlnotebook.ipynb** must detect and the
expected responses. Use it while modifying the notebook to keep behavior consistent across the
primary execution phases: configuration loading, data preparation, training, and export.

## Configuration parsing (`0. Load training configuration`, `0. Load dataset configuration`)

| Error case | User feedback | Remediation expectation | Notes |
| --- | --- | --- | --- |
| Hugging Face credentials missing when resolving `model.pretrained_model_name_or_path` from the Hub. | Raise a `RuntimeError` with guidance to login via `huggingface-cli login` or set `HUGGING_FACE_HUB_TOKEN`. Also emit a notebook cell log entry clarifying the check. | Manual fix. Notebook should stop before downloading models. | Add the guard before attempting `diffusers`/`huggingface_hub` downloads to avoid partial downloads. |
| Training or dataset TOML fails schema validation (type mismatch, missing keys). | Raise a `ValueError` describing the field and expected schema. Display a Markdown cell in the output summarizing the invalid entries. | Manual fix. Abort further cells to avoid cascading failures. | Keep validation logic co-located with the `tomli`/`pydantic` parsing utilities. |

## Data preparation (`1. Dataset discovery`, `2. Dataset setup`)

| Error case | User feedback | Remediation expectation | Notes |
| --- | --- | --- | --- |
| Caption/tag files with the configured extension (default `.txt`) are missing for discovered images. | Log each missing pair in the cell output (e.g., `warning` level via `logging`). After listing, raise a `FileNotFoundError` summarizing the total missing captions. | Manual fix. User must add captions or remove the images. | Ensure the scan runs before splitting validation prompts to prevent partial datasets. |
| Dataset checksum mismatches when verifying downloaded archives. | Raise a `RuntimeError` that includes the expected and actual hashes. | Manual fix or re-download prompt. Offer a suggestion to delete the corrupted archive. | Perform checksum verification immediately after download/unpack logic. |

## Training loop (`3. Optimizer setup`, `4. Training hyperparameters`, `5. Training`, `6. LoRA configuration`)

| Error case | User feedback | Remediation expectation | Notes |
| --- | --- | --- | --- |
| Insufficient VRAM detected while allocating model or during the first forward pass. | Catch `torch.cuda.OutOfMemoryError`, log a structured warning with suggested adjustments (reduce resolution, batch size, LoRA rank), then re-raise to stop the run. | Manual fix. Do not attempt automatic retry to avoid repeated failures. | Consider probing available GPU memory before constructing large tensors to provide proactive warnings. |

## Export and artifact packaging (`7. Validation preview`, `8. Save & export`)

| Error case | User feedback | Remediation expectation | Notes |
| --- | --- | --- | --- |
| Checkpoint checksum mismatches during final artifact packaging (e.g., when computing SHA256 before upload). | Log an error describing the mismatch and raise a `RuntimeError` to prevent publishing corrupted artifacts. | Manual fix—user should regenerate or re-export. | Ensure hashes are recomputed after moving files into the export directory. |

## General guidance

- Log messages should rely on the shared notebook logger configured near the top of the notebook so that they surface both in the cell output and saved logs.
- Raised exceptions must stop the cell execution, ensuring the user addresses the underlying issue before progressing.
- When documenting new sections or refactoring cells, reference this file and update the section names above if they change in the notebook.
