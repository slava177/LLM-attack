import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig,AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model
import warnings
import pandas as pd
from datasets import Dataset, DatasetDict
import os
from trl import RewardConfig, RewardTrainer
warnings.filterwarnings('ignore')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

warnings.filterwarnings('ignore')

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
# Load dataset
def load_dataset():
    df = pd.read_csv("./data/advbench/harmful_behaviors.csv")
    df = df.dropna(subset=["goal", "target"])  # Remove empty rows

    df["negative"] = "I'm sorry, but I can't assist with that request."

    # Split into 90% train, 10% validation
    train_df = df.sample(frac=0.9, random_state=42)
    val_df = df.drop(train_df.index)

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df)
    })
    return dataset["train"], dataset["validation"]


def format_dataset(examples):
    kwargs = {"padding": "max_length", "truncation": True, "max_length": 512, "return_tensors": "pt"}

    # Format chosen (preferred) and rejected (negative) responses
    prompt_plus_chosen_response = f"User: {examples['goal']}\nAssistant: {examples['target']}"
    prompt_plus_rejected_response = f"User: {examples['goal']}\nAssistant: {examples['negative']}"

    tokens_chosen = tokenizer(prompt_plus_chosen_response, **kwargs)
    tokens_rejected = tokenizer(prompt_plus_rejected_response, **kwargs)

    return {
        "input_ids_chosen": tokens_chosen["input_ids"].squeeze(), 
        "attention_mask_chosen": tokens_chosen["attention_mask"].squeeze(),
        "input_ids_rejected": tokens_rejected["input_ids"].squeeze(), 
        "attention_mask_rejected": tokens_rejected["attention_mask"].squeeze()
    }



# Load model and tokenizer
model_name = "distilroberta-base"
# model_name = "google/flan-t5-small"
model = AutoModelForSequenceClassification.from_pretrained(model_name, quantization_config=quantization_config, num_labels=1)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
#     model.config.pad_token_id = model.config.eos_token_id


# Load dataset
train_dataset, val_dataset = load_dataset()
train_dataset = train_dataset.map(format_dataset)
val_dataset = val_dataset.map(format_dataset)


# Configuring the training arguments
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,
    lora_alpha=64,
    lora_dropout=0.1,
)

# Save model
save_path = "./distilroberta-base-reward-model"
training_args = RewardConfig(
    # output_dir="./flan-t5-reward-model",
    output_dir=save_path,
    report_to=None,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=15,
    # max_steps = 1000
    # save_total_limit=2,
    # logging_dir="./logs",
)
# Loading the RewardTrainer from TRL
trainer = RewardTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    # peft_config=peft_config
)
trainer.train()



# save_path = "./flan-t5-reward-model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)