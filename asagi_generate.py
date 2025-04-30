'''

In this module, Asagi (a VLM) is used to generate a text that describes the input image based on a given prompt.

'''
import requests
import torch
import transformers
import pickle
import json
import tqdm
from PIL import Image
from transformers import AutoModel, AutoProcessor, GenerationConfig
from huggingface_hub import snapshot_download


def main(system_prompt, user_agent_prompts_base, revised_images):
    # Set random seed
    transformers.set_seed(0)

    # Download the pretrained models
    snapshot_download(
        repo_id="MIL-UT/Asagi-8B",
        local_dir="/gs/bs/tga-c-ird-lab/chen/considerate-robot/my_models/Asagi-8B",
        local_dir_use_symlinks=False
    )

    # Load the pretrained model
    model_path = "/gs/bs/tga-c-ird-lab/chen/considerate-robot/my_models/Asagi-8B"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Set generation_configuration
    generation_config = GenerationConfig(
        do_sample=True,
        num_beams=5,
        max_new_tokens=256,
        temperature=0.7,
        repetition_penalty=1.5
    )

    # Create prompt
    i = 0
    for user_agent_prompt_base, image in tqdm.tqdm(zip(user_agent_prompts_base, revised_images)):
        if i == 10:
            break
        i += 1
        '''
        ref_user_prompt = "【ユーザの状況】\n"
        ref_user_prompt = "\nこれは画像である。\n"
        ref_user_prompt += f"ユーザの発話: {user_agent_prompt_base['ref_uttr']}\n"
        ref_user_prompt += f"ユーザの位置: {user_agent_prompt_base['ref_position']}\n"
        ref_user_prompt += f"ユーザが手にしている物: {user_agent_prompt_base['ref_has']}\n"
        ref_user_prompt += f"ユーザの近くにある物: {user_agent_prompt_base['ref_near_objs']}"

        ref_agent_prompt = '\n'.join(user_agent_prompt_base['ref_answer'])
        ref_agent_prompt += "\n[回答終了]"

        user_prompt = "【ユーザの状況】\n"
        user_prompt += f"ユーザの発話: {user_agent_prompt_base['uttr']}\n"
        user_prompt += f"ユーザの位置: {user_agent_prompt_base['position']}\n"
        user_prompt += f"ユーザが手にしている物: {user_agent_prompt_base['has']}\n"
        user_prompt += f"ユーザの近くにある物: {user_agent_prompt_base['near_objs']}"
        user_prompt += f"### 指示:\n<image>\nこの画像を見て、行動カテゴリとその行動カテゴリに割り振られた番号だけを各行動カテゴリごとに改行して出力してください。\n\n### 応答:\n"

        # CoT
        cot = 0
        if cot:
            user_prompt += "日常生活の常識に基づいて、順を追って考え、答えを出してください。"

        prompt = system_prompt + '\n' + ref_user_prompt + '\n' + ref_agent_prompt + '\n' + user_prompt
        '''
        prompt = f"### 指示:\n<image>\nこの画像を見たとき、次の2つの動作の選択肢が与えられます。(a) 机を片付ける (b) コップを持ってくる。回答としては、どちらか1つの選択肢のみを出力してください。\n\n### 応答:\n"
        inputs = processor(
            text=prompt, images=[image], return_tensors="pt"
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

        # Do not print the prompt
        if "<image>" in prompt:
            prompt = prompt.replace("<image>", " ")
        generated_text = generated_text.replace(prompt, "")

        print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    with open('data/system_prompt_v1.txt') as f:
        system_prompt = f.read()
    with open('data/prompt.json', 'r') as f:
        user_agent_prompts_base = [json.loads(l.strip()) for l in f]
    with open('data/revised_images.pkl', 'rb') as f:
        revised_images = pickle.load(f)

    main(system_prompt, user_agent_prompts_base, revised_images)
