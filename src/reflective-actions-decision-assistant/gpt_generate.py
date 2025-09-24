"""
This module is to create a prompt as input of GPT-4o, and obtain output from LLM which is stored as pickle file.
"""
import json
import os
import my_utils
import time
import random
import glob
import base64
import openai
from openai import OpenAI
from dotenv import load_dotenv


# Load OPENAI_API_KEY
loaded = load_dotenv()
# Create a client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def jpg_to_data_uri(path: str) -> str:
    '''
    This function is used to convert *.jpg to urls which can be feed into GPT-4o.
    '''
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def main(system_message, user_agent_prompts_base, image_urls):
    n_scenarios = len(image_urls)
    running_time = 0
    outputs = []
    for idx, (user_agent_prompt_base, image_url) in enumerate(zip(user_agent_prompts_base, image_urls)):
        ref_user_prompt = "【ユーザの状況】\n"
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

        # CoT
        # user_prompt += "日常生活の常識に基づいて、順を追って考え、答えを出してください。"

        # conversation template (one shot)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": ref_user_prompt},
            {"role": "assistant", "content": ref_agent_prompt},
            
            #{"role": "user",
            #"content": [
            #    {"type": "text", "text": "以下の写真をご覧ください。"},
            #    {"type":"image_url", "image_url": {"url": f"{image_url}"}}
            #]},
            
            {"role": "user", "content": user_prompt}
        ]
        start_time = time.time()
        completion = client.chat.completions.create(
            model="gpt-4-0613",
            messages=messages,
            max_tokens=1900,
            temperature=0.0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop='[回答終了]'
        )
        end_time = time.time()
        running_time += end_time - start_time
        response = completion.choices[0].message.content.strip().split('\n')
        print(f"=== Example [{idx + 1}/{n_scenarios}] ===\n{response}\n")
        # Convert the current iteration output from LLM to a list
        outputs.append(response)

    my_utils.pickle_write(outputs, 'model_outputs/llms/gpt4/anno.pkl', overwrite=True)
    print(f'running time: {(running_time) / 60:.3f}min')


if __name__ == '__main__':
    with open('prompts/system_prompt_anno_img.txt') as f:
        system_message = f.read()
    with open('data/prompt.jsonl', "r") as f:
        user_agent_prompts_base = [json.loads(l.strip()) for l in f]
    # Load the revised images
    raw_image_urls = glob.glob('data/revised_images/*.jpg')
    image_urls = [jpg_to_data_uri(raw_image_url) for raw_image_url in raw_image_urls]

    ref_image_urls = random.sample(image_urls, k=len(image_urls))

    main(system_message, user_agent_prompts_base, image_urls)
    print('Done!')
