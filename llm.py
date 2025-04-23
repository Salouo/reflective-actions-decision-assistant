"""
This module is to create a prompt as input of LLM, and obtain output from LLM which is stored as pickle file.

gpt4o
"""


from openai import OpenAI
import json
import os
import my_utils
import time
import tqdm
from dotenv import load_dotenv


# Load OPENAI_API_KEY
loaded = load_dotenv()
# Create a client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def main():
    with open('revised_role_prompt.txt') as f:
        system_message = f.read()

    with open('data/prompt.json', "r") as f:
        user_agent_prompts_base = [json.loads(l.strip()) for l in f]

    outputs = []
    for user_agent_prompt_base in tqdm.tqdm(user_agent_prompts_base):
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

        # template
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": ref_user_prompt},
            {"role": "assistant", "content": ref_agent_prompt},
            {"role": "user", "content": user_prompt}
        ]

        completion = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=messages,
            max_tokens=1900,
            temperature=0.0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop='[回答終了]'
        )

        outputs.append(completion)

    my_utils.pickle_write(outputs, 'llm_outputs/gpt4o_20241120_HaveAction_1.pickle', overwrite=True)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('Done')
    print(f'実行時間: {(end_time - start_time) / 60:.3f}min')
