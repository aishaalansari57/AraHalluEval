import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse
from models import load_model_and_tokenizer, MODEL_MAP
from inference import get_response

# Task-specific files and prompts
datasets = {
    "summarization": "summ_annotations.csv",
    "qa": "QA_merged_predictions.csv"
}
cols = {
    "summarization": "text",
    "qa": "question_text"
}
prompts = {
    "summarization": "لخص النص الآتي في جملة واحدة فقط، وأجب باللغة العربية:\n\n{text}\n\nالملخص:",
    "qa": "اجب على السؤال التالي باللغة العربية:\n\n{text}\n\nالجواب:"
}

def main(model_key, task):
    assert model_key in MODEL_MAP, f"{model_key} not found in MODEL_MAP."

    # Load model/tokenizer/device
    model, tokenizer, device = load_model_and_tokenizer(model_key)

    # Load data
    dataset_name = datasets[task]
    col_name = cols[task]
    #df = pd.read_csv(dataset_name).head(5)  # ✅ Limit to 5 rows
    df = pd.read_csv(dataset_name)
    prompt_ar = prompts[task]

    # Setup output directory
    output_dir = Path(f"{task}_{model_key.replace(':', '_')}")
    output_dir.mkdir(exist_ok=True)

    responses = []
    #for idx, row in tqdm(df.head(5).iterrows(), total=5, desc=f"{model_key} (first 5)"):
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=model_key):
        text_input = row[col_name]
        prompt = prompt_ar.format(text=text_input)
        try:
            response = get_response(prompt, model, tokenizer, device, model_key=model_key, task=task)
        except Exception as e:
            response = f"ERROR: {e}"
        responses.append(response)

    # Save all results in a single CSV
    df["response"] = responses
    df.to_csv(output_dir / "results.csv", index=False, encoding="utf-8-sig")

    print(f"\n✅ Done! Saved output to: {output_dir / 'results.csv'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Arabic summarization or QA with any supported model.")
    parser.add_argument("--model", type=str, required=True, help="Model key from MODEL_MAP (e.g., openai:gpt-4o)")
    parser.add_argument("--task", type=str, choices=["summarization", "qa"], default="summarization")
    args = parser.parse_args()

    main(args.model, args.task)
