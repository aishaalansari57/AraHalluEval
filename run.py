import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse
from models import load_model_and_tokenizer, MODEL_MAP
from inference import get_response

datasets = {
    "summarization": "summarization_test.csv",
    "qa":"qa_test.csv"
}
cols = {
    "summarization": "text",
    "qa": "question_test"
}
prompts = {
    "summarization": "لخص النص الآتي في جملة واحدة فقط، وأجب باللغة العربية:\n\n{text}\n\nالملخص:",
    "qa": "اجب على السؤال التالي باللغة العربية:\n\n{text}\n\nالجواب:"
}

def main(model_key, task):
    assert model_key in MODEL_MAP, f"{model_key} not found in MODEL_MAP."

    model, tokenizer, device = load_model_and_tokenizer(model_key)

    dataset_name = datasets[task]
    col_name = cols[task]

    df = pd.read_csv(dataset_name)
    prompt_ar = prompts[task]
    output_dir = Path(f"{task}_{model_key}")
    output_dir.mkdir(exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=model_key):
        text_input = row[col_name]
        prompt = prompt_ar.format(text=text_input)
        response = get_response(prompt, model, tokenizer, device)
        with open(output_dir / f"{idx+1}.txt", "w", encoding="utf-8") as f:
            f.write(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run summarization with a specified model.")
    parser.add_argument("--model", type=str, required=True, help="Model name from MODEL_MAP (e.g., Jais-6.7b)")
    parser.add_argument("--task", type=str, default="summarization")
    args = parser.parse_args()

    main(args.model, args.task)
