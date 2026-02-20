import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"

print("Loading generator model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model.eval()


def generate_answer(query: str, context_chunks: list[str]) -> str:
    # Take only top 3 chunks to fit into model context
    context = "\n\n".join(context_chunks[:3])

    prompt = f"""
Answer the question strictly using the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    # IMPORTANT: DO NOT truncate here
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            do_sample=False,
            num_beams=4        # improves answer quality
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in decoded:
        decoded = decoded.split("Answer:")[-1].strip()

    return decoded
