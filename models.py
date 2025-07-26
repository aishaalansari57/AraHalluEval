from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_MAP = {
    "Jais-1.3B": "inceptionai/jais-family-1p3b",
    "Jais-2.7b": "inceptionai/jais-family-2p7b",
    "Jais-6.7b": "inceptionai/jais-family-6p7b",
    "Noon": "Naseej/noon-7b",
    "Allam": "ALLaM-AI/ALLaM-7B-Instruct-preview",
    "Fanar": "QCRI/Fanar-1-9B",  # use the exact HF path if different
    "Gemma": "google/gemma-3-4b-pt",
    "llama":"meta-llama/Meta-Llama-3-8B",
    "qwen2.5-1.5b":"Qwen/Qwen2.5-1.5B",
    "qwen2.5-3b":"Qwen/Qwen2.5-3B",
    "qwen2.5-7b":"Qwen/Qwen2.5-7B",
    "bloom-1.7b":"bigscience/bloom-1b7",
    "bloom-":"bigscience/bloom-3b",
    "bloom-":"bigscience/bloom-7b",

}

def load_model_and_tokenizer(model_key):
    model_name = MODEL_MAP[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), tokenizer, device
