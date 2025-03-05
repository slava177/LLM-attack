from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define model path (local or Hugging Face repo)
model_path = "./helper_deepseek"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# Move model to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Chat with your fine-tuned LLM (type 'exit' to stop):")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = chat(user_input)
    print("LLM:", response)
