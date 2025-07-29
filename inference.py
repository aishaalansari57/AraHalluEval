import torch
import re
import openai
import together
from models import MODEL_MAP

def get_response(prompt, model, tokenizer, device, model_key, task="summarization", debug=True):
    try:
        # OpenAI models
        if model_key.startswith("openai"):
            model_name = (
                "gpt-4o" if "gpt-4o" in model_key else
                "gpt-4" if "gpt-4" in model_key else
                "gpt-3.5-turbo"
            )
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            decoded = response["choices"][0]["message"]["content"].strip()

        # Together AI models
        elif model_key.startswith("together"):
            model_name = MODEL_MAP[model_key]
            response = together.Complete.create(
                prompt=prompt,
                model=model_name,
                max_tokens=200,
                temperature=0.0,
                repetition_penalty=1.2
            )
            decoded = response['choices'][0]['text'].strip()
            if debug:
                print(f"[Together AI response]: {decoded}")

        # Hugging Face models
        else:
            # Tokenize and move to device
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}  # ✅ Move to device correctly

            # Generate
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                max_new_tokens=200,
                min_length=inputs["input_ids"].shape[-1] + 4,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )

            decoded = tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]

        # Post-processing: extract relevant part
        for key in ["الملخص:", "الترجمة:", "الجواب:"]:
            if key in decoded:
                summary = decoded.split(key)[-1].strip()
                break
        else:
            summary = decoded.strip()

        if task == "summarization":
            summary = re.split(r"[.!؟]\s*", summary)[0].strip()

        return summary

    except Exception as e:
        return f"ERROR: {e}"
