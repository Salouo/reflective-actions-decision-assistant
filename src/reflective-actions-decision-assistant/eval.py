"""

This module is used to evaluate LLM's performance(top-k, plus-k) from output of LLM.

llm_outputs: a list of LLM's output

"""
import sys
import my_utils
import random
import statistics


def main(outputs):
    # candidate_actions_str[0]: バナナを持ってくる; shape: 40
    candidate_actions_str: list[str] = my_utils.get_texts_sep_by_new_line('data/actions_ja.txt')
    # candidate_actions_str_with_id[0]: [1]バナナを持ってくる
    candidate_actions_str_with_id: list[str] = [f'[{i+1}]{candidate_action_str}' for i, candidate_action_str in enumerate(candidate_actions_str)]
    # correct label set: labels_v2['at-least3'] for 355 scenarios; 
    labels = my_utils.pickle_read('data/labels_v2.pkl')
    # Load the remaining scenario indices
    revised_scenario_idxs = my_utils.pickle_read('data/revised_scenario_idxs.pkl')
    accepted_scenario_idxs = revised_scenario_idxs['revised']   # index of scenario accepted by crowdworkers -> revised based on the original scenario set
    # Extract 355 correct label sets corresponding to each scenario from original labels
    # labels['at-least3'] indicates that, for each scenario, only actions that received at least three agreeing votes are selected as the correct labels.
    # shape: 355
    correct_label_set_list: list[list[int]] = [labels['at-least3'][scenario_idx] for scenario_idx in accepted_scenario_idxs]

    random.seed(0)
    rankings = []
    predictions = []


    for idx, output in enumerate(outputs):
        '''
        `output` is a list including predicted actions for a specific scenario.     e.g.: ['[6]ペットボトルを持ってくる', '[9]お菓子を持ってくる']
        '''
        prediction = []

        # If model predicts there is no appropriate action, append null into prediction list
        if ('該当なし' in output) or ('[該当なし]' in output):
            assert len(output) == 1
            predictions.append(prediction)

        # If model predicts appropriate actions exist, append them into prediction list
        else:
            # assert len(output) >= 1, output
            # Iterate over each predicted action in predicted actions set for a specific scenario
            for action_with_id in output:
                pred_cnt_each = 0
                # Find indices of predicted actions
                for id, candidate_action_str in enumerate(candidate_actions_str):
                    # Match each candiate action with the predicted actions
                    if candidate_action_str in action_with_id:
                        pred_cnt_each += 1      # one-to-one match
                        prediction.append(id)
                # assert pred_cnt_each == 1, action_with_id
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
    # print(mean,'±', std)

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
    outputs_path = "/gs/bs/tga-c-ird-lab/chen/reflective-action-decision-assistant/model_outputs/llms/gpt4/anno-only.pkl"
    outputs = my_utils.pickle_read(outputs_path)
    main(outputs)
