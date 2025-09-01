import subprocess
import json

def generate_answer(prompt, docs, model="llama3"):
    """Generate an answer using Ollama + retrieved docs"""
    context = "\n".join(docs)
    final_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer clearly:"

    cmd = ["ollama", "run", model, final_prompt]
    result = subprocess.run(cmd, capture_output=True, text=True)

    return result.stdout.strip()

