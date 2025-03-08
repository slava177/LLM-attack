import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
import warnings
import pandas as pd
from datasets import Dataset, DatasetDict
import os
from trl import RewardConfig, RewardTrainer

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
# Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.special_tokens['<extra_0>']
model.config.pad_token_id = tokenizer.pad_token_id

# Load dataset
def load_dataset():
    df = pd.read_csv("./data/advbench/harmful_behaviors.csv")
    df = df.dropna(subset=["goal", "target"])  # Remove empty rows

    df["negative"] = "I'm sorry. I can not assist you with this request."

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

def format_dataset(examples):
    kwargs = {"padding": "max_length", "truncation": True, "max_length": 512, "return_tensors": "pt"}
    # eos_token = tokenizer.eos_token if tokenizer.eos_token else "<|endoftext|>"
    prompt_plus_chosen_response = examples["goal"] + "\n" + examples["target"]
    prompt_plus_rejected_response = examples["goal"] + "\n" + examples["negative"]
    tokens_chosen = tokenizer(prompt_plus_chosen_response, **kwargs)
    tokens_rejected = tokenizer(prompt_plus_rejected_response, **kwargs)
    return {
        "input_ids_chosen": tokens_chosen["input_ids"].squeeze(), "attention_mask_chosen": tokens_chosen["attention_mask"].squeeze(),
        "input_ids_rejected": tokens_rejected["input_ids"].squeeze(), "attention_mask_rejected": tokens_rejected["attention_mask"].squeeze()
    }

def tokenize_function(example):
    tokenized_inputs = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512,return_attention_mask=True)
    tokenized_inputs["labels"] = [
        -100 if token == tokenizer.pad_token_id else token for token in tokenized_inputs["input_ids"]
    ]
    return tokenized_inputs

# Apply LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Correct type for language models
    inference_mode=False,
    r=16,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    bias="none"
)


# Apply LoRA to the model
model = get_peft_model(model, peft_config)
model.config.pad_token_id = tokenizer.pad_token_id
#model.gradient_checkpointing_enable()

# Load and preprocess dataset
train_dataset, val_dataset = load_dataset()
train_dataset = train_dataset.map(format_dataset)
val_dataset = val_dataset.map(format_dataset)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

# Configuring the training arguments
save_path = "./helper_deepseek32"
training_args = RewardConfig(
    output_dir=save_path,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=15,
    learning_rate=1e-5,
    bf16=True,  # Use BF16 for better memory efficiency
    #deepspeed="zero3.json",  # Offload to CPU (need to create zero3.json)
)

# Use Trainer instead of RewardTrainer
trainer = RewardTrainer(
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