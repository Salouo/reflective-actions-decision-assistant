import argparse
from io import BytesIO

import requests
import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (get_model_name_from_path,
                            process_images, tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


disable_torch_init()

model_checkpoint_path = "llm-jp/llm-jp-3-vila-14b"
model_name = get_model_name_from_path(model_checkpoint_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_checkpoint_path, model_name)

image_path = "path/to/image"
image_files = [
    image_path
]
images = load_images(image_files)

query = "<image>\nこの画像について説明してください。"

conv_mode = "llmjp_v3"
conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], query)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=[
            images_tensor,
        ],
        do_sample=False,
        num_beams=1,
        max_new_tokens=256,
        use_cache=True,
    )

outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
print(outputs)
