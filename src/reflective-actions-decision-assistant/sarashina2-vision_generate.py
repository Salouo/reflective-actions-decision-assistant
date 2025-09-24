import json
import pickle
import my_utils
import time
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def main(system_prompt, user_agent_prompts_base, revised_images):
    # Conuting running time
    running_time = 0
    # Define model path
    model_path = "sbintuitions/sarashina2-vision-14b"
    cache_dir = "/gs/bs/tga-c-ird-lab/chen/considerate-robot/models_cache/sarashina2-vision-14b"

    # Load model and processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_dir,)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True
    )

    outputs = []
    n_senarios = len(revised_images)
    for idx, (user_agent_prompt_base, image) in enumerate(zip(user_agent_prompts_base, revised_images)):
        ref_user_prompt = "【ユーザの状況】\n"
        ref_user_prompt += f"ユーザの発話: {user_agent_prompt_base['ref_uttr']}\n"
        ref_user_prompt += f"ユーザの位置: {user_agent_prompt_base['ref_position']}\n"
        ref_user_prompt += f"ユーザが手にしている物: {user_agent_prompt_base['ref_has']}\n"
        ref_user_prompt += f"ユーザの近くにある物: {user_agent_prompt_base['ref_near_objs']}"

        ref_agent_prompt =  "### Assistant:\n"
        ref_agent_prompt += '\n'.join(user_agent_prompt_base['ref_answer'])
        ref_agent_prompt += "\n[回答終了]\n\n"

        user_prompt = "### Human:\n"
        user_prompt += "【ユーザの状況】\n"
        user_prompt += f"ユーザの発話: {user_agent_prompt_base['uttr']}\n"
        user_prompt += f"ユーザの位置: {user_agent_prompt_base['position']}\n"
        user_prompt += f"ユーザが手にしている物: {user_agent_prompt_base['has']}\n"
        user_prompt += f"ユーザの近くにある物: {user_agent_prompt_base['near_objs']}"

        prompts = system_prompt + '\n'
        prompts += ref_user_prompt + '\n'
        prompts += ref_agent_prompt + '\n'
        prompts += user_prompt

        message = [{"role": "user", "content": prompts}]
        text_prompt = processor.apply_chat_template(message, add_generation_prompt=True)
        """text_prompt: <s><|prefix|><|file|><|suffix|>A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.

        ### Human: `user_prompt`
        ### Assistant:"""
        start_time = time.time()

        inputs = processor(
            text=[text_prompt],
            ### Input image or not
            images=[image.convert('RGB')],
            # images=None,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")
        stopping_criteria = processor.get_stopping_criteria(["\n###"])

        # Inference: Generation of the output
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.0,
            do_sample=False,
            stopping_criteria=stopping_criteria,
        )
        generated_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        end_time = time.time()
        running_time += end_time - start_time

        # Obtain the response from model
        response = output_text[0]

        # Sometimes the last character of responses is `[`.
        if response.endswith('['):
            response = response[:-1]
        response = response.strip().split('\n')
        # Sometimes responses are not followed by [回答終了].
        if response[-1] == "[回答終了]": 
            response = response[:-1]
        # Print the output at current iteration
        print(f"=== Example [{idx + 1}/{n_senarios}] ===\n{response}\n")
        # Collect the output at current iteration
        outputs.append(response)
    
    my_utils.pickle_write(outputs, 'model_outputs/vlms/sarashina2-vision/anno-and-img.pkl', overwrite=True)
    print(f'running time: {(running_time) / 60:.3f}min')


if __name__ == "__main__":
    with open('prompts/system_prompt_anno.txt') as f:
        system_prompt = f.read()
    with open('data/prompt.jsonl', 'r') as f:
        user_agent_prompts_base = [json.loads(l.strip()) for l in f]
    with open('data/revised_images.pkl', 'rb') as f:
        revised_images = pickle.load(f)
    
    main(system_prompt, user_agent_prompts_base, revised_images)
