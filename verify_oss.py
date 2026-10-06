"""Quick verification that open-source models work with LangChain.

Run:  ./.venv/bin/python verify_oss.py
"""
import os

print("=" * 60)
print("1) OLLAMA (Llama 3.2 3B via langchain_ollama)")
print("=" * 60)
from langchain_ollama import OllamaLLM

ollama_llm = OllamaLLM(model="llama3.2:3b")
resp = ollama_llm.invoke("Reply with exactly: OLLAMA-WORKS")
print("Ollama response:", resp.strip())

print()
print("=" * 60)
print("2) HUGGING FACE (transformers pipeline via langchain_huggingface)")
print("=" * 60)
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=32,
    device_map="cpu",
)
hf_llm = HuggingFacePipeline(pipeline=pipe)
hf_resp = hf_llm.invoke("Reply with exactly: HF-WORKS")
print("HF response:", hf_resp.strip())

print()
print("ALL CHECKS PASSED ✔")