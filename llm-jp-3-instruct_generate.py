import torch
import json
import pickle
import time
import my_utils
from transformers import AutoTokenizer, AutoModelForCausalLM


def main(system_prompt, user_agent_prompts_base):
    running_time = 0

    cache_dir = "/gs/bs/tga-c-ird-lab/chen/considerate-robot/models_cache/llm-jp-13b"
    model_path = "llm-jp/llm-jp-3-13b-instruct"

    tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-3-13b-instruct")
    model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir, device_map="auto", torch_dtype=torch.bfloat16)

    outputs = []
    n_senarios = len(user_agent_prompts_base)
    for idx, user_agent_prompt_base in enumerate(user_agent_prompts_base):
        ref_user_prompt = "【ユーザの状況】\n"
        ref_user_prompt += f"ユーザの発話: {user_agent_prompt_base['ref_uttr']}\n"
        ref_user_prompt += f"ユーザの位置: {user_agent_prompt_base['ref_position']}\n"
        ref_user_prompt += f"ユーザが手にしている物: {user_agent_prompt_base['ref_has']}\n"
        ref_user_prompt += f"ユーザの近くにある物: {user_agent_prompt_base['ref_near_objs']}"

        ref_agent_prompt = '\n'.join(user_agent_prompt_base['ref_answer'])
        ref_agent_prompt += "\n[回答終了]\n\n"

        user_prompt = "【ユーザの状況】\n"
        user_prompt += f"ユーザの発話: {user_agent_prompt_base['uttr']}\n"
        user_prompt += f"ユーザの位置: {user_agent_prompt_base['position']}\n"
        user_prompt += f"ユーザが手にしている物: {user_agent_prompt_base['has']}\n"
        user_prompt += f"ユーザの近くにある物: {user_agent_prompt_base['near_objs']}"

        prompts = system_prompt + '\n'
        prompts += ref_user_prompt + '\n'
        prompts += ref_agent_prompt + '\n'
        prompts += user_prompt

        chat = [
            {"role": "system", "content": "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"},
            {"role": "user", "content": prompts},
        ]
        tokenized_input = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(model.device)

        start_time = time.time()
        
        # ① 直接得到 Tensor；不要再取 ["input_ids"]
        input_ids = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(model.device)          # input_ids: shape [1, L]

        input_len = input_ids.size(1)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=128,
                do_sample=False,
                top_p=0.95,
                temperature=0.0,
                repetition_penalty=1.05,
            )

        end_time = time.time()
        running_time += end_time - start_time

        # Obtain the model's answer
        gen_ids = output_ids[0, input_len:]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        response = response.strip().split('\n')

        # Sometimes responses are not followed by [回答終了].
        if response[-1] == "[回答終了]": 
            response = response[:-1]

        outputs.append(response)
        print(f"=== Example [{idx + 1}/{n_senarios}] ===\n{response}\n")

    my_utils.pickle_write(outputs, 'model_outputs/llms/llm-jp-3-instruct-13b/anno-only.pkl', overwrite=True)
    print(f'running time: {(running_time) / 60:.3f}min')


if __name__ == "__main__":
    with open('prompts/system_prompt_anno.txt') as f:
        system_prompt = f.read()
    with open('data/prompt.jsonl', 'r') as f:
        user_agent_prompts_base = [json.loads(l.strip()) for l in f]

    main(system_prompt, user_agent_prompts_base)
