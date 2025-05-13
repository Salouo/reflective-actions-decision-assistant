import torch
import json
from huggingface_hub import snapshot_download
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image



local_llava = snapshot_download(
    repo_id="llava-hf/llama3-llava-next-8b-hf",
    local_dir="/gs/bs/tga-c-ird-lab/chen/considerate-robot/pretrained_models/llava3-llava-next-8b-hf",
    local_dir_use_symlinks=False
)

processor = LlavaNextProcessor.from_pretrained(local_llava, trust_remote_code=True)
model = LlavaNextForConditionalGeneration.from_pretrained(local_llava, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True) 

# prepare image and text prompt, using the appropriate prompt template
img_path = "data/raw_images/01_01.jpg"
image = Image.open(img_path)

with open('prompts/system_prompt_en.txt', 'r') as f:
  system_prompt = f.read()
with open('data/prompt_test.jsonl', 'r') as f:
  user_agent_prompts_base = [json.loads(l.strip()) for l in f]

prompt_base = user_agent_prompts_base[0]

ref_user_prompt = "【ユーザの状況】\n"
ref_user_prompt = "\nこれは画像である。\n"
ref_user_prompt += f"ユーザの発話: {prompt_base['ref_uttr']}\n"
ref_user_prompt += f"ユーザの位置: {prompt_base['ref_position']}\n"
ref_user_prompt += f"ユーザが手にしている物: {prompt_base['ref_has']}\n"
ref_user_prompt += f"ユーザの近くにある物: {prompt_base['ref_near_objs']}"
ref_user_prompt += f"### 応答:\n"
ref_agent_prompt = '\n'.join(prompt_base['ref_answer'])
ref_agent_prompt += "\n[回答終了]"

user_prompt = "【ユーザの状況】\n"
user_prompt += f"ユーザの発話: {prompt_base['uttr']}\n"
user_prompt += f"ユーザの位置: {prompt_base['position']}\n"
user_prompt += f"ユーザが手にしている物: {prompt_base['has']}\n"
user_prompt += f"ユーザの近くにある物: {prompt_base['near_objs']}"

# Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
  {

    "role": "user",
    "content": [
        {"type": "text", "text": f"{system_prompt}\n\n{ref_user_prompt}\n\n{ref_agent_prompt}\n\n{user_prompt}\n"},
        {"type": "image"},
      ],
  },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=100)

print(f'Answer: {processor.decode(output[0], skip_special_tokens=True)}')
