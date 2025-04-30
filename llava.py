import torch
import requests
from huggingface_hub import snapshot_download
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image



local_llava = snapshot_download(
    repo_id="llava-hf/llama3-llava-next-8b-hf",
    local_dir="/gs/bs/tga-c-ird-lab/chen/considerate-robot/my_models/llava3-llava-next-8b-hf",
    local_dir_use_symlinks=False
)

processor = LlavaNextProcessor.from_pretrained(local_llava, trust_remote_code=True)
model = LlavaNextForConditionalGeneration.from_pretrained(local_llava, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True) 

# prepare image and text prompt, using the appropriate prompt template
img_path = "raw_images/01_01.jpg"
image = Image.open(img_path)

# Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "If you are a robot, which action will you take when you see this scenario?\n \
            (a)Bring a cup. (b)Clear the table. You should only output a character (a or b) as your answer."},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=100)

print(processor.decode(output[0], skip_special_tokens=True))
