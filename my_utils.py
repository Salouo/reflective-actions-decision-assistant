import pickle
import torch
from sklearn.metrics import classification_report
import random
import numpy as np
import os
import statistics
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN


def pickle_read(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def pickle_write(obj, file, overwrite=False):
    if overwrite:
        with open(file, 'wb') as f:
            pickle.dump(obj, f)
    else:
        with open(file, 'xb') as f:
            pickle.dump(obj, f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eval_topk(topk_predictions, correct_label_set_list, k, is_plusk=False):
    '''
    This function is used to compute top-k and plus-k.

    topk_predictions: (355, 40)
    correct_label_set_list: (355, number of predicted actions for each corresponding scenario)
    k: number of predicted actions we consider in rankings
    is_plusk: top-k or plus-k
    '''

    n = k   # 1, 3, 5
    predictions_size = len(topk_predictions)    # 355
    assert predictions_size == len(correct_label_set_list)  # 355
    if predictions_size != 355:
        print('Alert! input_size:', predictions_size)
    correct_label_cnt_list = []
    isin_list = []
    recall_list = []
    correct_pred_cnt_list = []

    # Iterate over each scenario
    for idx in range(predictions_size):

        # For each scenario, deduplicate its corresponding action indices and obtain the number of label actions for each scenario.
        correct_label_set = set(correct_label_set_list[idx])    # set of indices of correct actions for a specific scenario

        correct_label_cnt = len(correct_label_set)  # number of correct actions for a specific scenario

        correct_label_cnt_list.append(correct_label_cnt)    # correct_label_cnt_list: (355, number of correct actions(unique) for each corresponding scenario)

        # If compute plus-k metrics?
        if is_plusk:
            n = correct_label_cnt + k

        assert n <= len(topk_predictions[idx]), print(n, len(topk_predictions[idx]))
        # Select k indices of predicted action from ranking for a specific scenario
        topk_prediction = set(topk_predictions[idx][:n])
        # For each scenario, take the intersection of the predicted actions and the label actions.
        intersection = topk_prediction & correct_label_set
        correct_pred_cnt = len(intersection)    # length of correctly predicted actions corresponding to a specific scenario
        correct_pred_cnt_list.append(correct_pred_cnt)  # list containing number of correctly predicted actions for all scenarios

        # Create isin score list for all scenarios
        isin_value = int(correct_pred_cnt > 0)
        isin_list.append(isin_value)

        # Create recall score list for all scenarios
        recall = correct_pred_cnt / correct_label_cnt
        recall_list.append(recall)

    is_in = sum(isin_list) / predictions_size   # final is_in score
    macro_recall = sum(recall_list) / predictions_size   # final macro recall score (average recall rate per scenario)
    micro_recall = sum(correct_pred_cnt_list) / sum(correct_label_cnt_list)     # micro_recall
    
    return {'is_in':is_in, 'macro_recall':macro_recall, 'micro_recall':micro_recall}


def get_texts_sep_by_new_line(src_path):
    texts = []
    with open(src_path) as f:
        doc = f.read()
    doc = doc.rstrip()
    contents = doc.split('\n')
    for content in contents:
        if content!='':
            texts.append(content)
    return texts

def regularize_response(response: str) -> str:
    # Sometimes the last character of responses is `[`.
    if response.endswith('['):
        response = response[:-1]
    response = response.strip().split('\n')
    # Sometimes responses are not followed by [回答終了].
    if response[-1] == "[回答終了]": 
        response = response[:-1]
    # Return the regularized response
    return response

