from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Define model path (local or Hugging Face repo)
model_path = "./helper_deepseek32"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set device (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Chatbot is ready! Type 'exit' to end the conversation.\n")

# Chat loop
while True:
    user_input = input("You: ")  # Get user input
    
    if user_input.lower() == "exit":  # Exit condition
        print("Chatbot: Goodbye!")
        break

    # Tokenize input
    inputs = tokenizer(user_input, return_tensors="pt").to(device)

    # Generate response
    outputs = model.generate(inputs.input_ids, max_new_tokens=150)

    # Decode and print response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Chatbot: {response}\n")
