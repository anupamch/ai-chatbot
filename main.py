from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers  import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import uvicorn
app = FastAPI()

# Load FAQ
with open("faq.txt", "r") as f:
    faq_lines = f.readlines()

# Initialize RAG
embedder = SentenceTransformer('all-MiniLM-L6-v2')
faq_embeddings = embedder.encode(faq_lines, convert_to_tensor=True)
dimension = faq_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(faq_embeddings.cpu().numpy())

# Load LLM
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
model = AutoModelForCausalLM.from_pretrained('distilgpt2')

# Pydantic model for request
class Question(BaseModel):
    question: str

# RAG function
async def get_answer(question: str) -> str:
    q_embedding = embedder.encode([question], convert_to_tensor=True)
    D, I = index.search(q_embedding.cpu().numpy(), k=1)
    context = faq_lines[I[0][0]]
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("Answer:")[1].strip()

# API endpoint
@app.post("/chat")
async def chat(question: Question):
    if not question.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    answer = await get_answer(question.question)
    return {"answer": answer}

# Run directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)