import gradio as gr
import pandas as pd
from pathlib import Path
import ast

# --------------------
# File Paths
# --------------------
BASE = Path(__file__).parent
CSV_PATH = BASE / "Combined_QA_inferences.csv"
REVIEW_CSV = BASE / "qa_hallucination_annotations_reviewed.csv"

# --------------------
# Config
# --------------------
FACTORS = [
    "Named-Entity Hallucination",
    "Temporal/Number Hallucination",
    "Factual Contradiction",
    "Conflict Hallucination",
    "Knowledge Source Conflict",  # ✅ Newly added factor
    "Grammar Hallucination",
    "Generic/Imprecise Hallucination",
    "Instruction Inconsistency",
    "Code-Switching"
]
YES_NO = ["No", "Yes"]

ALL_MODELS = [
    "Allam", "Fanar", "Gemma", "Jais-6.7b", "Noon",
    "bloom-7b", "llama", "qwen2.5-7b",
    "Maverick", "DeepSeek-v3", "DeepSeek-r1",
    "Qwq", "GPT-4o", "GPT-o3"
]

# --------------------
# Load & Prepare Data
# --------------------
csv_df = pd.read_csv(CSV_PATH)

# Convert gold answer lists to string
def extract_gold_text(val):
    try:
        return ', '.join([ans['text'] for ans in ast.literal_eval(val)])
    except:
        return str(val)

csv_df['gold_text'] = csv_df['answers'].apply(extract_gold_text)
sample_indices = list(range(len(csv_df)))

# --------------------
# Functions
# --------------------
def get_record(idx):
    row = csv_df.iloc[int(idx)]
    question = row["question_text"]
    gold = row["gold_text"]
    outputs = []

    for model in ALL_MODELS:
        ans = row.get(model, "N/A")
        outputs.append({
            "model": model,
            "answer": ans,
            "annotations": ["No"] * len(FACTORS),
            "comments": ""
        })
    return question, gold, outputs


def save_review(idx, *values):
    idx = int(idx)
    records = []

    for i, model in enumerate(ALL_MODELS):
        offset = i * (len(FACTORS) + 1)
        ann = values[offset:offset + len(FACTORS)]
        comment = values[offset + len(FACTORS)]
        row = csv_df.iloc[idx]

        record = {
            "sample_index": idx,
            "question": row["question_text"],
            "gold_answer": row["gold_text"],
            "model": model,
            "answer": row.get(model, "")
        }
        for j, f in enumerate(FACTORS):
            record[f] = ann[j]
        record["Comments"] = comment
        records.append(record)

    # Save to file
    if REVIEW_CSV.exists():
        df_out = pd.read_csv(REVIEW_CSV)
    else:
        df_out = pd.DataFrame(columns=["sample_index", "question", "gold_answer", "model", "answer"] + FACTORS + ["Comments"])

    df_out = df_out[~((df_out["sample_index"] == idx) & (df_out["model"].isin(ALL_MODELS)))]
    df_out = pd.concat([df_out, pd.DataFrame(records)], ignore_index=True)
    df_out.to_csv(REVIEW_CSV, index=False, encoding="utf-8-sig")
    return f"✅ Saved {len(records)} model reviews for sample {idx}"


def update_ui(idx_val):
    q, g, outs = get_record(idx_val)
    updates = [q, g]
    for i, (a_box, radio_list, c_box) in enumerate(model_blocks):
        updates.append(outs[i]["answer"])
        updates.extend(outs[i]["annotations"])
        updates.append(outs[i]["comments"])
    return updates

# --------------------
# Gradio UI
# --------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 QA Hallucination Review Interface (Arabic Questions)")

    idx = gr.Slider(minimum=0, maximum=len(csv_df) - 1, step=1, label="Sample Index")
    q_box = gr.Textbox(label="السؤال (Question)", interactive=False)
    g_box = gr.Textbox(label="الإجابة الذهبية (Gold Answer)", interactive=False)

    model_blocks = []
    inputs = []

    for model in ALL_MODELS:
        with gr.Group():
            gr.Markdown(f"### {model}")
            a = gr.Textbox(label="Model Answer", lines=3, interactive=False)
            radios = [gr.Radio(YES_NO, value="No", label=factor) for factor in FACTORS]
            comment = gr.Textbox(label="Comment", lines=2)
            model_blocks.append((a, radios, comment))
            inputs.extend(radios + [comment])

    save_btn = gr.Button("💾 Save")
    status = gr.Textbox(label="Status", interactive=False)

    idx.change(fn=update_ui, inputs=[idx], outputs=[q_box, g_box] + [x for b in model_blocks for x in (b[0], *b[1], b[2])])
    save_btn.click(fn=save_review, inputs=[idx] + inputs, outputs=[status])

    gr.Markdown("> Run with `python qa_review.py`")

if __name__ == "__main__":
    demo.launch(share=False)
