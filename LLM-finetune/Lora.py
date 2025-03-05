import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from trl import RewardConfig, RewardTrainer
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from datasets import Dataset



def load_dataset():

    # Load CSV
    df = pd.read_csv("./data/advbench/harmful_behaviors.csv")

    # Ensure it has the correct columns: "question" and "response"
    df = df.dropna(subset=["goal", "target"])  # Remove empty rows if any

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    return dataset


def format_dataset(example):
    return {"text": f"User: {example['question']}\nAssistant: {example['response']}"}

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

# Load model and tokenizer
# model_name = "distilroberta-base"
model_name = "deepseek-ai/DeepSeek-V3"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model.config.pad_token_id = tokenizer.pad_token_id
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
#     model.config.pad_token_id = model.config.eos_token_id


# Load dataset
dataset = load_dataset()
dataset = dataset.map(format_dataset)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

save_path = "./helper_deepseek"
# Configuring the training arguments
training_args = RewardConfig(
    output_dir=save_path,
    report_to=None,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=15,
)
# Loading the RewardTrainer from TRL
trainer = RewardTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=tokenized_dataset,
    # eval_dataset=val_dataset
)
trainer.train()


# Save model
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
