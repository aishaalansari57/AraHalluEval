import torch
import re

def get_response(prompt, model, tokenizer, device):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    input_len = input_ids.shape[-1]

    output_ids = model.generate(
        input_ids,
        temperature=0.0,            # Fully deterministic (greedy decoding)
        top_p=1.0,                  # No nucleus sampling
        do_sample=False,           # No random sampling → crucial for reproducibility
        max_new_tokens=200,
        min_length=input_len + 4,
        repetition_penalty=1.2     # Optional, can keep
    )


    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    if any(key in decoded for key in ["الملخص:", "الترجمة:", "الجواب:"]):
        for key in ["الملخص:", "الترجمة:", "الجواب:"]:
            if key in decoded:
                summary = decoded.split(key)[-1].strip()
                break
    else:
        summary = decoded.strip()

    if task=="summarization":# إذا أردت فقط أول جملة
        summary = re.split(r"[.!؟]\s*", summary)[0].strip()
        return summary
    else:
      return summary

