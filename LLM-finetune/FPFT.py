import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
import warnings
import pandas as pd
from datasets import Dataset, load_dataset
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

warnings.filterwarnings('ignore')

model_name = "deepseek-ai/deepseek-llm-7b-base"
# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16  # Use float16 for faster computation
)
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=bnb_config, 
    device_map="auto"
)
# Apply LoRA for memory-efficient fine-tuning
lora_config = LoraConfig(
    r=8,  # Low-rank adaptation size
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention layers
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("✅ DeepSeek LLM Loaded with LoRA and 4-bit Precision!")

dataset = load_dataset("imdb")

def tokenize_function(examples):
    inputs = tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )
    inputs["labels"] = inputs["input_ids"].copy()
    
    # Replace padding tokens in labels with -100 to ignore them in loss computation
    inputs["labels"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels]
        for labels in inputs["labels"]
    ]
    return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)
# Subset the dataset for faster experimentation
tokenized_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(500))
tokenized_val_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))

# Configuring the training arguments
save_path = "./helper_deepseekR1"
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-4,  # Lower learning rate for LoRA fine-tuning
    per_device_train_batch_size=1,  # Reduce batch size for memory efficiency
    gradient_accumulation_steps=8,  # Simulate larger batch size
    num_train_epochs=0.5,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    fp16=True,  # Mixed precision training
)

# Use Trainer instead of RewardTrainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer
)
print(f"start training \n")
model.train()
trainer.train()

# Save model
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model and tokenizer saved at {save_path}")