import time
import pickle
import json
import my_utils
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


def main(system_prompt, user_agent_prompts_base, revised_images):
    # Conuting running time
    running_time = 0

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

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

        image = image.convert('RGB')

        messages = [
            {
                "role": "user",
                "content": [
                    #{
                    #    "type": "image",
                    #    "image": image,
                    #},
                    {"type": "text", "text": prompts},
                ],
            }
        ]

        start_time = time.time()

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            #images=None,
            videos=video_inputs,
            #videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        end_time = time.time()
        running_time += end_time - start_time

        # Obtain the response from model
        response = output_text[0]
        
        # Regularize the raw response (it sometimes includes useless characters)
        re_response = my_utils.regularize_response(response)

        # Print the output at current iteration
        print(f"=== Example [{idx + 1}/{n_senarios}] ===\n{re_response}\n")
        # Collect the output at current iteration
        outputs.append(re_response)

    my_utils.pickle_write(outputs, 'model_outputs/vlms/qwen-vl-7b-instruct/anno-and-img-2.pkl', overwrite=True)
    print(f'running time: {(running_time) / 60:.3f}min')


if __name__ == "__main__":
    with open('prompts/system_prompt_anno.txt') as f:
        system_prompt = f.read()
    with open('data/prompt.jsonl', 'r') as f:
        user_agent_prompts_base = [json.loads(l.strip()) for l in f]
    with open('data/revised_images.pkl', 'rb') as f:
        revised_images = pickle.load(f)
    
    main(system_prompt, user_agent_prompts_base, revised_images)
