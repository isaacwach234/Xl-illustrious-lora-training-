# Xl-illustrious-lora-training-

A LoRA trainer notebook for training Illustrious LoRAs.

## Documentation
- [Lambda Stack 22.04 arm64 environment setup](docs/setup.md)
- [Notebook error handling expectations](docs/error-handling.md)

## Configuration overview

Two TOML files under `project_name/configs/` provide reusable defaults for the notebook:

- `training_config.example.toml` captures run metadata, optimizer options, and LoRA-specific knobs.
- `dataset_config.example.toml` describes where image/caption pairs live and how validation previews are rendered.

Copy the examples, rename them (for example `training_config.toml`), and point the "0. Load training configuration" and "0. Load dataset configuration" cells in **Illustriousxlnotebook.ipynb** at your customized files.

### Training configuration parameters

| Parameter | Description | Acceptable values | Notebook widget/cell |
| --- | --- | --- | --- |
| `experiment.run_name` | Human-readable label for the session and prefix for output folders. | Any ASCII string; keep under 64 characters to avoid filesystem limits. | "1. Run metadata" → **Run name** text box. |
| `experiment.output_dir` | Directory where checkpoints, weights, and logs are stored. | Relative or absolute path; must be writable. | "1. Run metadata" → **Output directory** picker. |
| `experiment.seed` | Random seed applied to Python, NumPy, and PyTorch. | Non-negative integer; commonly 0–9999. | "1. Run metadata" → **Seed** numeric field. |
| `model.pretrained_model_name_or_path` | Base SDXL model to fine-tune. | Hugging Face repo ID or local path. | "2. Model preparation" → **Base model** dropdown/input. |
| `optimizer.mixed_precision` | Precision mode for training. | `"bf16"`, `"fp16"`, or `"no"`. | "3. Optimizer setup" → **Mixed precision** selector. |
| `optimizer.use_8bit_adam` | Toggle for bitsandbytes AdamW. | `true` or `false`. | "3. Optimizer setup" → **Use 8-bit Adam** switch. |
| `training.train_batch_size` | Images processed per optimizer step before accumulation. | Integer 1–4 (GPU VRAM dependent). | "4. Training hyperparameters" → **Batch size** field. |
| `training.gradient_accumulation_steps` | Gradient accumulation multiplier. | Integer 1–16; combined with batch size for effective batch. | "4. Training hyperparameters" → **Grad accumulation** field. |
| `training.learning_rate` | Initial learning rate for AdamW. | Float 5e-5–1e-3 typical for LoRA. | "4. Training hyperparameters" → **Learning rate** slider. |
| `training.lr_scheduler` | Learning rate scheduler strategy. | `"constant"`, `"cosine"`, or `"cosine_with_restarts"`. | "4. Training hyperparameters" → **Scheduler** dropdown. |
| `training.lr_warmup_steps` | Steps before scheduler starts decaying LR. | Integer ≥0; 50–500 recommended. | "4. Training hyperparameters" → **Warmup steps** field. |
| `training.max_train_epochs` | Number of dataset passes. | Integer ≥1; 5–15 common for 100 images. | "4. Training hyperparameters" → **Epochs** field. |
| `training.checkpointing_steps` | Interval for saving checkpoints. | Integer ≥50; align with dataset size. | "5. Checkpointing" → **Save every N steps** field. |
| `training.dataloader_num_workers` | Parallel workers for data loading. | Integer 0–8 depending on CPU threads. | "4. Training hyperparameters" → **DataLoader workers** field. |
| `lora.network_rank` | Rank of LoRA decomposition. | Integer 4–64; higher consumes more VRAM. | "6. LoRA configuration" → **Network rank** slider. |
| `lora.network_alpha` | Scaling factor applied to LoRA layers. | Integer or float, typically 0.5–2× rank. | "6. LoRA configuration" → **Network alpha** field. |

### Dataset configuration parameters

| Parameter | Description | Acceptable values | Notebook widget/cell |
| --- | --- | --- | --- |
| `dataset.root` | Base directory for the dataset. | Existing directory path. | "2. Dataset setup" → **Dataset root** picker. |
| `dataset.train_folder` | Subfolder holding training pairs. | Relative path inside `root`. | "2. Dataset setup" → **Training folder** text field. |
| `dataset.caption_extension` | Required caption suffix matching images. | `.txt`, `.json`, etc.; defaults to `.txt`. | "2. Dataset setup" → **Caption extension** selector. |
| `dataset.image_extensions` | Allowed image formats. | List of lowercase extensions (e.g., `.jpg`). | "2. Dataset setup" → **Image extensions** multi-select. |
| `dataset.max_train_images` | Limit on processed pairs. | Integer ≥1; set to dataset size to include all. | "2. Dataset setup" → **Max images** numeric input. |
| `dataset.resolution` | Target square resolution for crops/resizes. | Integer multiple of 64; SDXL defaults to 1024. | "2. Dataset setup" → **Resolution** field. |
| `dataset.shuffle_before_validation` | Shuffle pairs before reserving validation prompts. | `true` or `false`. | "2. Dataset setup" → **Shuffle before validation** toggle. |
| `validation.prompt` | Positive prompt for preview renders. | Any prompt string using trigger token. | "7. Validation preview" → **Prompt** textarea. |
| `validation.negative_prompt` | Negative prompt for previews. | Any prompt string; can be empty. | "7. Validation preview" → **Negative prompt** textarea. |
| `validation.num_images` | Number of images to generate during preview. | Integer 1–8 recommended. | "7. Validation preview" → **Images to render** field. |
| `concept.trigger_token` | Token inserted in prompts to call the concept. | Custom string such as `<sks-person>`. | "6. LoRA configuration" → **Trigger token** field. |
| `concept.prior_loss_weight` | Weight for prior preservation loss. | Float 0.0–1.0. | "6. LoRA configuration" → **Prior loss weight** slider. |

## Dataset layout and validation

Place your training assets inside `dataset/train/` as matched `.jpg` (or `.png`) images with corresponding caption files using the same stem and the extension defined in your dataset configuration. For example:

```
dataset/
  train/
    subject_0001.jpg
    subject_0001.txt
    subject_0002.jpg
    subject_0002.txt
    ...
```

The notebook's "2. Dataset setup" cell scans `dataset.train_folder`, confirms that every image listed has a caption with the same filename stem, and warns you in-line if any pairs are missing. Only validated pairs contribute toward the `dataset.max_train_images` count before training begins. Validation preview prompts can optionally live under `dataset/validation/` if you want to manage them alongside your training data; update the TOML paths accordingly.

Keep filenames alphanumeric with underscores to avoid shell parsing issues, and ensure captions are UTF-8 encoded plain text.
