import torch
import re
import openai
import together
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os

# Hugging Face login (use token with access to gated models)
#login(token="")
# Add OpenAI and Together API keys
#openai.api_key = ""
#together.api_key  = ""

MODEL_MAP = {
    # Hugging Face models
    "Jais-6.7b": "inceptionai/jais-family-6p7b",
    "Noon": "Naseej/noon-7b",
    "Allam": "ALLaM-AI/ALLaM-7B-Instruct-preview",
    "Fanar": "QCRI/Fanar-1-9B",
    "Gemma": "google/gemma-3-4b-pt",
    "llama": "meta-llama/Meta-Llama-3-8B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "bloom-7b": "bigscience/bloom-7b",

    # OpenAI models
    "openai:gpt-4o": "openai",
    "openai:gpt-o3": "openai",


    # Together AI models
    "together:deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "together:deepseek-r1": "deepseek-ai/DeepSeek-R1-0528",
    "together:qwen-qwq": "Qwen/QwQ-32B",
    "together:llama4-Maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
}


def load_model_and_tokenizer(model_key):
    if model_key.startswith("openai") or model_key.startswith("together"):
        return None, None, None  # No local model/tokenizer needed
    model_name = MODEL_MAP[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), tokenizer, device
