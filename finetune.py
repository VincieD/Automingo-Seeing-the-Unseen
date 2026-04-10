from datasets import load_dataset, Image
from trl import SFTTrainer, SFTConfig
from transformers import Qwen3VLForConditionalGeneration, BitsAndBytesConfig
import torch
from peft import LoraConfig, PeftModel
from transformers import AutoProcessor, AutoModelForImageTextToText

# dataset_name = "trl-lib/llava-instruct-mix"
# structure: images, prompt, completion
dataset_name = "ibarcelo/Automingo_dataset"

train_dataset = load_dataset(dataset_name)

# If dataset has splits
if isinstance(train_dataset, dict):
    train_dataset = train_dataset["train"]

image_cols = [f"image_{i}" for i in range(1, 6)]

# Ensure image columns decode to PIL
for col in image_cols:
    if col in train_dataset.column_names:
        train_dataset = train_dataset.cast_column(col, Image())

def expand_to_single_image_rows(batch):
    new_images = []
    new_prompts = []
    new_completions = []

    batch_size = len(batch["question"])

    for i in range(batch_size):
        question = batch["question"][i].strip()
        answer = (batch.get("ground_truth_answer")[i] or "").strip()
        reasoning = (batch.get("ground_truth_reasoning")[i] or "").strip()

        if reasoning:
            completion_text = (
                f"{answer}\n\nReasoning:\n{reasoning}" if answer else f"Reasoning:\n{reasoning}"
            )
        else:
            completion_text = answer

        # iterate over the 5 images and create one row per image
        for col in image_cols:
            img = batch[col][i]
            if img is None or img == "":
                continue

            new_images.append([img])  # list with single image

            new_prompts.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                }
            ])

            new_completions.append([
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": completion_text}
                    ],
                }
            ])

    return {
        "images": new_images,
        "prompt": new_prompts,
        "completion": new_completions,
    }

train_dataset = train_dataset.map(
    expand_to_single_image_rows,
    batched=True,
    remove_columns=train_dataset.column_names,
)

print(train_dataset)


model_name = "Qwen/Qwen3-VL-8B-Instruct" # "Qwen/Qwen3-VL-8B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype="float32",
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,                        # Load the model in 4-bit precision to save memory
        bnb_4bit_compute_dtype=torch.float16,     # Data type used for internal computations in quantization
        bnb_4bit_use_double_quant=True,           # Use double quantization to improve accuracy
        bnb_4bit_quant_type="nf4"                 # Type of quantization. "nf4" is recommended for recent LLMs
    )
)


# You may need to update `target_modules` depending on the architecture of your chosen model.
# For example, different VLMs might have different attention/projection layer names.
peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
)

output_dir = "Qwen3-VL-8B-Instruct-Automingo"

# Configure training arguments using SFTConfig
# https://github.com/huggingface/trl/blob/main/trl/trainer/sft_config.py
training_args = SFTConfig(
    # Training schedule / optimization
    num_train_epochs=1,
    #max_steps=10,                                         # Number of dataset passes. For full trainings, use `num_train_epochs` instead
    per_device_train_batch_size=8,                        # Batch size per GPU/CPU
    gradient_accumulation_steps=8,                        # Gradients are accumulated over multiple steps → effective batch size = 4 * 8 = 32
    warmup_steps=5,                                       # Gradually increase LR during first N steps
    learning_rate=2e-4,                                   # Learning rate for the optimizer
    optim="adamw_8bit",                                   # Optimizer
    max_length=None,                                      # For VLMs, truncating may remove image tokens, leading to errors during training. max_length=None avoids it
    # loss_type="nll",

    # Logging / reporting
    output_dir=output_dir,                                # Where to save model checkpoints and logs
    logging_steps=1,                                      # Log training metrics every N steps
    report_to="wandb",                                # Experiment tracking tool

    # Hub integration
    push_to_hub=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")

# trainer.save_model(output_dir)
# trainer.push_to_hub(dataset_name=dataset_name)

# 1) Save adapter (this is what SFTTrainer/PEFT produces)
trainer.save_model(output_dir)  # saves adapter_config.json + adapter_model.safetensors

# 2) Reload base in fp16 (NOT 4-bit) for merging
base = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# 3) Load adapter onto base, merge, unload
merged = PeftModel.from_pretrained(base, output_dir)
merged = merged.merge_and_unload()

# 4) Save full model package (this creates config.json)
merged.save_pretrained(output_dir, safe_serialization=True)

# 5) Save processor/tokenizer too (critical for VL)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
processor.save_pretrained(output_dir)

# 6) Push to hub (now it’s a real standalone model repo)
trainer.push_to_hub()