import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, TaskType
import warnings
import pandas as pd
from datasets import Dataset, DatasetDict

warnings.filterwarnings('ignore')

# Load dataset
def load_dataset():
    df = pd.read_csv("./data/advbench/harmful_behaviors.csv")
    df = df.dropna(subset=["goal", "target"])  # Remove empty rows

    # Split into 90% train, 10% validation
    train_df = df.sample(frac=0.9, random_state=42)
    val_df = df.drop(train_df.index)

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df)
    })
    return dataset["train"], dataset["validation"]

def format_dataset(example):
    return {"text": f"User: {example['goal']}\nAssistant: {example['target']}"}

def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

# Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-V3"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model.config.pad_token_id = tokenizer.pad_token_id

# Load and preprocess dataset
train_dataset, val_dataset = load_dataset()
train_dataset = train_dataset.map(format_dataset)
val_dataset = val_dataset.map(format_dataset)

tokenized_train_dataset = train_dataset.map(lambda e: tokenize_function(e, tokenizer), batched=True)
tokenized_val_dataset = val_dataset.map(lambda e: tokenize_function(e, tokenizer), batched=True)

# LoRA Configuration for Causal LM
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Correct type for DeepSeek-V3
    inference_mode=False,
    r=16,
    lora_alpha=64,
    lora_dropout=0.1,
)

# Configuring the training arguments
save_path = "./helper_deepseek"
training_args = TrainingArguments(
    output_dir=save_path,
    report_to=None,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=15,
)

# Use Trainer instead of RewardTrainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer
)
trainer.train()

# Save model
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
