from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_MAP = {
    "Jais-1.3B": "inceptionai/jais-family-1p3b",
    "Jais-2.7b": "inceptionai/jais-family-2p7b",
    "Jais-6.7b": "inceptionai/jais-family-6p7b",
    "Noon-7b": "aseej/noon-7b",
    "Allam": "ALLaM-AI/ALLaM-7B-Instruct-preview",
    "Fanar-1.9B": "QCRI/Fanar-1-9B",  # use the exact HF path if different
    "Gemma-3.4B": "google/gemma-3-4b-pt"
}

def load_model_and_tokenizer(model_key):
    model_name = MODEL_MAP[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), tokenizer, device
