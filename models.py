import torch
import re
import openai
import together
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add OpenAI and Together API keys
openai.api_key = "your-openai-api-key"
together.api_key = "your-together-api-key"

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
    "openai:gpt-3.5": "openai",
    "openai:gpt-4": "openai",
    "openai:gpt-4o": "openai",

    # Together AI models
    "together:deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "together:deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "together:qwen-qwq": "qwen/Qwen-QwQ-32B",
    "together:llama4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct-FP8",
    "together:llama4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
}


def load_model_and_tokenizer(model_key):
    if model_key.startswith("openai") or model_key.startswith("together"):
        return None, None, None  # No local model/tokenizer needed
    model_name = MODEL_MAP[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), tokenizer, device
