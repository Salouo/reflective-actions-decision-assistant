'''

In this module, Asagi (a VLM) is used to generate a text that describes the input image based on a given prompt.

'''


import requests
import torch
import transformers
from PIL import Image
from transformers import AutoModel, AutoProcessor, GenerationConfig


def main():
    transformers.set_seed(42)
    model_path = "MIL-UT/Asagi-4B"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    generation_config = GenerationConfig(
        do_sample=True,
        num_beams=5,
        max_new_tokens=256,
        temperature=0.7,
        repetition_penalty=1.5
    )

    prompt = ("以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n"
                "### 指示:\n<image>\nこの画像を見て、次の質問に詳細かつ具体的に答えてください。この写真はどこで撮影されたものか教えてください。また、画像の内容についても詳しく説明してください。\n\n### 応答:\n")

    sample_image_path = "./input_image_asagi/01_09.jpg"
    image = Image.open(sample_image_path)

    inputs = processor(
        text=prompt, images=image, return_tensors="pt"
    )
    inputs_text = processor.tokenizer(prompt, return_tensors="pt")
    inputs['input_ids'] = inputs_text['input_ids']
    inputs['attention_mask'] = inputs_text['attention_mask']
    for k, v in inputs.items():
        if v.dtype == torch.float32:
            inputs[k] = v.to(model.dtype)
    inputs = {k: inputs[k].to(model.device) for k in inputs if k != "token_type_ids"}

    generate_ids = model.generate(
        **inputs,
        generation_config=generation_config
    )
    generated_text = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # do not print the prompt
    if "<image>" in prompt:
        prompt = prompt.replace("<image>", " ")
    generated_text = generated_text.replace(prompt, "")

    print(f"Generated text: {generated_text}")

    # >>> Generated text:  この写真は東京の渋谷で撮影されたものです。夜の渋谷の街並みが写っており、高層ビルが立ち並び、街灯やネオンサインが輝いています。


if __name__ == "__main__":
    main()

