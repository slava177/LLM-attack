import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
import warnings
import pandas as pd
import os
from tqdm import tqdm
from accelerate import Accelerator
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
warnings.filterwarnings('ignore')

# Initialize Accelerator
accelerator = Accelerator()

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
# Load model in 8bit
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"  # Let accelerate handle device placement
)
model = prepare_model_for_kbit_training(model)

# Apply LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Ensure tokenizer pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.special_tokens['<extra_0>']
model.config.pad_token_id = tokenizer.pad_token_id

# ===========================
# 🔹 Custom Dataset Loader
# ===========================
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data.iloc[idx]

        # Prepare input prompt
        prompt_plus_chosen_response = f"User: {example['goal']}\nAssistant: {example['target']}"
        prompt_plus_rejected_response = f"User: {example['goal']}\nAssistant: {example['negative']}"

        kwargs = {"padding": "max_length", "truncation": True, "max_length": 128, "return_tensors": "pt"}
        tokens_chosen = tokenizer(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer(prompt_plus_rejected_response, **kwargs)

        return {
            "input_ids_j": tokens_chosen["input_ids"].squeeze(),
            "attention_mask_j": tokens_chosen["attention_mask"].squeeze(),
            "input_ids_k": tokens_rejected["input_ids"].squeeze(),
            "attention_mask_k": tokens_rejected["attention_mask"].squeeze(),
        }

# ===========================
# 🔹 Load Dataset & DataLoader
# ===========================
def load_dataset():
    df = pd.read_csv("./data/advbench/harmful_behaviors.csv")
    df = df.dropna(subset=["goal", "target"])

    df["negative"] = "I'm sorry, but I can't assist with that request."

    # Split into 90% train, 10% validation
    train_df = df.sample(frac=0.9, random_state=42)
    val_df = df.drop(train_df.index)

    return train_df, val_df

# Load datasets
train_df, val_df = load_dataset()

# Create PyTorch datasets
train_dataset = CustomDataset(train_df)
val_dataset = CustomDataset(val_df)

# Define PyTorch DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# ===========================
# 🔹 Custom RewardTrainer
# ===========================
class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Move inputs to model device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]

        # Compute ranking loss
        loss = -torch.nn.functional.logsigmoid(rewards_j - rewards_k).mean()

        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss

# ===========================
# 🔹 Training Configuration
# ===========================
save_path = "./helper_deepseek32"
training_args = TrainingArguments(
    output_dir=save_path,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=20,
    learning_rate=1e-5,
    bf16=True,
    report_to=None
)

# Initialize Trainer with DataLoader
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Keep track of dataset
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# ===========================
# 🔹 Training Loop
# ===========================
print("Start training")

trainer.create_optimizer()
for epoch in range(15):  # num_train_epochs
    for batch in tqdm(train_dataloader):
        # Move batch to GPU
        batch = {k: v.to(model.device) for k, v in batch.items()}
        # Forward pass
        loss = trainer.compute_loss(model, batch)
        loss.backward()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()

print("Training complete!")

# Save model
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

print("Model and tokenizer saved!")
