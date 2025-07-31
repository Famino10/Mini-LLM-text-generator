from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# 1️⃣ Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")

# 2️⃣ Load dataset
def load_dataset(tokenizer, file_path):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )

train_dataset = load_dataset(tokenizer, "data.txt")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 3️⃣ Training configuration
training_args = TrainingArguments(
    output_dir="./mini-llm",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=1
)

# 4️⃣ Trainer
trainer = Trainer(
    model=model,  # ✅ Now model is defined
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

# 5️⃣ Train
trainer.train()

# 6️⃣ Save fine-tuned model
trainer.save_model("./mini-llm")
tokenizer.save_pretrained("./mini-llm")
print("✅ Training complete and model saved!")
