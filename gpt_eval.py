"""

This module is used to evaluate LLM's performance(top-k, plus-k) from output of LLM.

llm_outputs: a list of LLM's output

"""


import sys
sys.path.append('my_utils.py')
import my_utils
import openai
import random
import statistics


def main(llm_outputs):
    # candidate_actions_str[0]: バナナを持ってくる
    candidate_actions_str = my_utils.get_texts_sep_by_new_line('data/actions_ja.txt')
    # candidate_actions_str_with_id[0]: [1]バナナを持ってくる
    candidate_actions_str_with_id = [f'[{i+1}]{candidate_action_str}' for i, candidate_action_str in enumerate(candidate_actions_str)]

    # correct label set: labels_v2['at-least3'] for 355 scenarios
    labels = my_utils.pickle_read('data/labels_v2.pickle')

    revised_scenario_idxs = my_utils.pickle_read('data/revised_scenario_idxs.pickle')
    accepted_scenario_idxs = revised_scenario_idxs['revised']   # index of scenario accepted by crowdworkers -> revised based on the original scenario set
    correct_label_set_list = [labels[f'at-least3'][scenario_idx] for scenario_idx in accepted_scenario_idxs]    # correct label set list corresponding to each scenario


    random.seed(42)
    rankings = []
    predictions = []

    for llm_output in llm_outputs:

        generate = llm_output.choices[0].message.content    # list containing answer from LLM for a specific scenario; e.g. ['[2]...', '[36]...']
        # Return a list that includes predicted actions for a specific scenario
        generate = generate.strip().split('\n')     # e.g.: ['[6]ペットボトルを持ってくる', '[9]お菓子を持ってくる']
        prediction = []

        # If LLM predicts there is no appropriate action, append null into prediction list
        if ('該当なし' in generate) or ('[該当なし]' in generate):
            assert len(generate) == 1
            predictions.append(prediction)

        # If LLM predicts appropriate actions exist, append them into prediction list
        else:
            assert len(generate) >= 1, generate
            # Iterate over each predicted action in predicted actions set for a specific scenario
            for action_with_id in generate:
                # print(len(action_with_id))
                # print(action_with_id)
                pred_cnt_each = 0
                # Find indices of predicted actions
                for id, candidate_action_str in enumerate(candidate_actions_str):
                    # Match each candiate action with predicted actions
                    if candidate_action_str in action_with_id:
                        pred_cnt_each += 1      # number of legal predicted action +1
                        prediction.append(id)
                assert pred_cnt_each == 1, action_with_id
            # 为了取平均top1？
            random.shuffle(prediction)
            predictions.append(prediction)

        # print(len(prediction))
        ranking = [i for i in range(40) if i not in prediction]
        random.shuffle(ranking)

        # Create topk list (但这样是否很不准确？首先topk现在并不是按概率排的，而且一旦k选择超过llm输出的回答的数量，就会有无关答案。。)
        # list + list: [a, b] + [c, d, f] = [a, b, c, d, f]
        ranking = prediction + ranking

        assert len(ranking) == 40

        # rankings: a list includes 355 lists containing indices of topk answers for each scenario
        rankings.append(ranking)


    # average and variance of number of predicted actions for a specific scenario
    cnt = [len(prediction) for prediction in predictions]
    mean = statistics.mean(cnt)
    std = statistics.pstdev(cnt)
    print(mean,'±', std)

    # Evaluate is_in, macro-recall, and micro-recall across top-1, top-3, top-5
    for k in [1, 3, 5]:
        # rankings: (355, 40)
        # correct_label_set_list: (355, number of predicted actions for each scenario)
        topk_metrics = my_utils.eval_topk(rankings, correct_label_set_list, k=k)

        for metrics, v in topk_metrics.items():
            if 'is_in' in metrics :
                print(f'top_{k}_is_in: {v:.5f}')

    # Evaluate is_in, macro-recall, and micro-recall plus-1, plus-3, plus-5
    for k in [0, 1, 3, 5]:
        plusk_metrics = my_utils.eval_topk(rankings, correct_label_set_list, k=k, is_plusk=True)
        for metrics, v in plusk_metrics.items():
            if 'is_in' in metrics :
                print(f'plus_{k}_is_in: {v:.5f}')


if __name__ == "__main__":
    llm_outputs = my_utils.pickle_read('llm_outputs/gpt4o_20241120_HaveAction_1.pickle')
    print(f'Model: {llm_outputs[0].model}')
    main(llm_outputs)
