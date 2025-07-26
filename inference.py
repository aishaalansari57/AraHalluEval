import torch
import re
import openai
import together

def get_response(prompt, model, tokenizer, device, model_key, task="summarization"):
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
        model_name = model_key.split(":")[1]
        response = together.Complete.create(
            prompt=prompt,
            model=model_name,
            max_tokens=200,
            temperature=0.0,
            repetition_penalty=1.2
        )
        decoded = response['output']['choices'][0]['text'].strip()

    # Hugging Face local models
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        input_len = input_ids.shape[-1]

        output_ids = model.generate(
            input_ids,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            max_new_tokens=200,
            min_length=input_len + 4,
            repetition_penalty=1.2
        )

        decoded = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

    # Post-processing: extract relevant portion
    for key in ["الملخص:", "الترجمة:", "الجواب:"]:
        if key in decoded:
            summary = decoded.split(key)[-1].strip()
            break
    else:
        summary = decoded.strip()

    if task == "summarization":
        summary = re.split(r"[.!؟]\s*", summary)[0].strip()

    return summary
