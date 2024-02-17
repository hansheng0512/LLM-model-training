from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Load the fine-tuned model
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")

# Input text for generation
input_text = "One day"

# Tokenize input text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate text
output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

# Decode and print generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text:", generated_text)
