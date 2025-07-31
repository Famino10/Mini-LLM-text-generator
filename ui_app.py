import gradio as gr

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# 1️⃣ Load fine-tuned model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("./mini-llm")
model = GPT2LMHeadModel.from_pretrained("./mini-llm")

# 2️⃣ Text generation function
def generate_text(prompt, max_len, temp):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones_like(input_ids)
    
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_len,
        do_sample=True,
        temperature=temp,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # ✅ Decode to normal text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


# 3️⃣ Create Gradio UI
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Enter your prompt"),
        gr.Slider(20, 200, step=10, value=50, label="Max Length"),
        gr.Slider(0.1, 1.5, step=0.1, value=0.7, label="Creativity (Temperature)")
    ],
    outputs="text",
    title="Mini LLM Text Generator",
    description="Type a prompt and let your fine-tuned GPT model generate text."
)

iface.launch()
