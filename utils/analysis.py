import os

import editdistance
import numpy as np


def analyse_observations_and_predictions(predicted_actions_per_frame, observed_actions_per_frame,
                                         unobserved_actions_per_frame, action_to_id, save_path, save_file_name):
    observed_actions, observed_lengths = aggregate_actions_and_lengths(observed_actions_per_frame)
    unobserved_actions, unobserved_lengths = aggregate_actions_and_lengths(unobserved_actions_per_frame)
    predicted_actions, predicted_lengths = aggregate_actions_and_lengths(predicted_actions_per_frame)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    mof = compute_mof(predicted_actions_per_frame, unobserved_actions_per_frame)
    save_file_name_mof = 'mof_' + str(round(mof, 2)).ljust(4, '0') + '_' + save_file_name
    with open(os.path.join(save_path, save_file_name_mof), mode='w') as f:
        f.write('Observed\n')
        for observed_action, observed_length in zip(observed_actions, observed_lengths):
            f.write('\t' + observed_action.ljust(24, ' ') + '\t' + str(observed_length).rjust(4, ' ') + '\n')
        f.write('\nUnobserved\n')
        for unobserved_action, unobserved_length in zip(unobserved_actions, unobserved_lengths):
            f.write('\t' + unobserved_action.ljust(24, ' ') + '\t' + str(unobserved_length).rjust(4, ' ') + '\n')
        f.write('\nPredicted\n')
        for predicted_action, predicted_length in zip(predicted_actions, predicted_lengths):
            f.write('\t' + predicted_action.ljust(24, ' ') + '\t' + str(predicted_length).rjust(4, ' ') + '\n')

    moc, correct_per_class, wrong_per_class = compute_moc(predicted_actions_per_frame, unobserved_actions_per_frame,
                                                          action_to_id)
    save_file_name_moc = 'moc_' + str(round(moc, 2)).ljust(4, '0') + '_' + save_file_name
    id_to_action = {action_id: action for action, action_id in action_to_id.items()}
    with open(os.path.join(save_path, save_file_name_moc), mode='w') as f:
        f.write(' '.rjust(30, ' ') + '\tCorr.:\tWrong:\tAcc.:\n')
        for action_id in range(len(id_to_action)):
            action = id_to_action[action_id]
            correct, wrong = correct_per_class[action_id], wrong_per_class[action_id]
            if correct + wrong > 0:
                acc = round(correct / (correct + wrong), 2)
                f.write(str(action_id).rjust(3, ' ') + ' | ' + action.rjust(24, ' ') + '\t' +
                        str(int(correct)) + '\t' + str(int(wrong)) + '\t' + str(acc).ljust(4, '0') + '\n')


def analyse_full_split_moc(moc, correct_per_class, wrong_per_class, action_to_id, save_path):
    save_file_name = 'moc_full_split_' + str(round(moc, 2)).ljust(4, '0') + '.txt'
    id_to_action = {action_id: action for action, action_id in action_to_id.items()}
    total_num_frames = 0
    with open(os.path.join(save_path, save_file_name), mode='w') as f:
        f.write(' '.rjust(30, ' ') + '\tCorr.:\tWrong:\tAcc.:\n')
        for action_id in range(len(id_to_action)):
            action = id_to_action[action_id]
            correct, wrong = correct_per_class[action_id], wrong_per_class[action_id]
            if correct + wrong > 0:
                total_num_frames += correct + wrong
                acc = round(correct / (correct + wrong), 2)
                f.write(str(action_id).rjust(3, ' ') + ' | ' + action.rjust(24, ' ') + '\t' +
                        str(int(correct)) + '\t' + str(int(wrong)) + '\t' + str(acc).ljust(4, '0') + '\n')
        f.write('\nTotal number of frames: ' + str(int(total_num_frames)) + '\n')


def analyse_performance_per_future_action(predicted_actions_per_video, unobserved_actions_per_video,
                                          transition_action_per_video, save_path):
    performances = {}
    for predicted_actions, unobserved_actions, transition_action in zip(predicted_actions_per_video,
                                                                        unobserved_actions_per_video,
                                                                        transition_action_per_video):
        transition_action_is_not_finished = transition_action == unobserved_actions[0]
        _, unobserved_lengths = aggregate_actions_and_lengths(unobserved_actions)
        unobserved_actions_initial_frames = [0] + list(accumulate(unobserved_lengths))
        for action_idx, (initial_frame, final_frame) in enumerate(zip(unobserved_actions_initial_frames[:-1],
                                                                      unobserved_actions_initial_frames[1:])):
            unobserved = unobserved_actions[initial_frame:final_frame]
            predicted = predicted_actions[initial_frame:final_frame]
            accuracy = np.mean(np.array(unobserved) == np.array(predicted)).item()
            if transition_action_is_not_finished:
                performances.setdefault(action_idx, []).append(accuracy)
            else:
                performances.setdefault(action_idx + 1, []).append(accuracy)
    performances = {action_idx: (sum(perf_list) / len(perf_list), len(perf_list))
                    for action_idx, perf_list in performances.items()}
    suffixes = {1: 'st', 2: 'nd', 3: 'rd'}
    save_file_name = 'mfap_full_split.txt'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, save_file_name), mode='w') as f:
        f.write('Future Actions Average Accuracy:\n')
        for action_idx in range(200):
            perf = performances.get(action_idx)
            if perf is None:
                break
            fa_avg_acc, num_videos = perf
            fa_avg_acc = round(fa_avg_acc, 4)
            suffix = suffixes.get(action_idx)
            suffix = 'th' if suffix is None else suffix
            prefix = (str(action_idx) + suffix + ': ').rjust(6)
            f.write(prefix + str(fa_avg_acc).ljust(6, '0') + ' | ' + str(num_videos).rjust(3, ' ') + '\n')


def compute_mof(predicted_actions_per_frame, unobserved_actions_per_frame):
    paf, uaf = np.array(predicted_actions_per_frame), np.array(unobserved_actions_per_frame)
    accuracy = np.mean(paf == uaf).item()
    return accuracy


def compute_moc(predicted_actions, unobserved_actions, action_to_id):
    num_classes = len(action_to_id)
    correct_per_class = np.zeros(num_classes, dtype=np.float32)
    wrong_per_class = np.zeros(num_classes, dtype=np.float32)
    for predicted_action, unobserved_action in zip(predicted_actions, unobserved_actions):
        if predicted_action == unobserved_action:
            correct_per_class[action_to_id[unobserved_action]] += 1
        else:
            wrong_per_class[action_to_id[unobserved_action]] += 1
    moc = 0
    n = 0
    for cpc, wpc in zip(correct_per_class, wrong_per_class):
        if cpc + wpc > 0:
            moc += cpc / (cpc + wpc)
            n += 1
    return moc / n, correct_per_class, wrong_per_class


def aggregate_actions_and_lengths(actions_per_frame):
    """Identify the actions in a video and count how many frames they last for.

    Given a list of actions (e.g. ['a', 'a', 'a', 'b', 'b']) summarise the actions and count their lifespan. For
    the example input just given, the function returns (['a', 'b'], [3, 2]).

    Arg(s):
        actions_per_frame - list containing the actions per frame.
    Returns:
        A tuple containing two lists. The first list contains the actions that happened throughout the video in
        sequence, whereas the second list contains the count of frames for which each action lived for.
    """
    actions, lengths = [], []
    if not actions_per_frame:
        return actions, lengths
    start_idx = 0
    for i, action in enumerate(actions_per_frame):
        if action != actions_per_frame[start_idx]:
            actions.append(actions_per_frame[start_idx])
            lengths.append(i - start_idx)
            start_idx = i
    actions.append(actions_per_frame[start_idx])
    lengths.append(len(actions_per_frame) - start_idx)
    return actions, lengths


def accumulate(lst):
    total = 0
    new_list = []
    for item in lst:
        total += item
        new_list.append(total)
    return new_list


def compute_segmental_edit_score_single_video(unobserved_actions_per_frame, predicted_actions_per_frame):
    normalised_levenshtein_distance = editdistance.eval(unobserved_actions_per_frame, predicted_actions_per_frame) * 1.0
    normalised_levenshtein_distance /= len(unobserved_actions_per_frame)  # predicted has the same length
    return 1 - normalised_levenshtein_distance


def compute_segmental_edit_score_multiple_videos(unobserved_actions_per_video, predicted_actions_per_video):
    seg_edit_scores = []
    for unobserved_actions_per_frame, predicted_actions_per_frame in zip(unobserved_actions_per_video,
                                                                         predicted_actions_per_video):
        seg_edit_score = compute_segmental_edit_score_single_video(unobserved_actions_per_frame,
                                                                   predicted_actions_per_frame)
        seg_edit_scores.append(seg_edit_score)
    return np.array(seg_edit_scores).mean()


def overlap_f1_single_video(unobserved_actions_per_frame, predicted_actions_per_frame, action_to_id, num_classes,
                            bg_class, overlap):
    true_intervals = np.array(segment_intervals(unobserved_actions_per_frame))
    true_labels = np.array(aggregate_actions_and_lengths(unobserved_actions_per_frame)[0])
    pred_intervals = np.array(segment_intervals(predicted_actions_per_frame))
    pred_labels = np.array(aggregate_actions_and_lengths(predicted_actions_per_frame)[0])

    # Remove background labels
    # if bg_class is not None:
    #     true_intervals = true_intervals[true_labels != bg_class]
    #     true_labels = true_labels[true_labels != bg_class]
    #     pred_intervals = pred_intervals[pred_labels != bg_class]
    #     pred_labels = pred_labels[pred_labels != bg_class]

    n_true = true_labels.shape[0]
    n_pred = pred_labels.shape[0]

    # We keep track of the per-class TPs, and FPs.
    # In the end we just sum over them though.
    TP = np.zeros(num_classes, dtype=np.float)
    FP = np.zeros(num_classes, dtype=np.float)
    true_used = np.zeros(n_true, dtype=np.float)

    for j in range(n_pred):
        # Compute IoU against all others
        intersection = np.minimum(pred_intervals[j, 1], true_intervals[:, 1]) - np.maximum(pred_intervals[j, 0],
                                                                                           true_intervals[:, 0])
        union = np.maximum(pred_intervals[j, 1], true_intervals[:, 1]) - np.minimum(pred_intervals[j, 0],
                                                                                    true_intervals[:, 0])
        IoU = (intersection.astype(np.float) / union) * (pred_labels[j] == true_labels)

        # Get the best scoring segment
        idx = IoU.argmax()

        # If the IoU is high enough and the true segment isn't already used
        # Then it is a true positive. Otherwise is it a false positive.
        action_id = action_to_id[pred_labels[j]]
        if IoU[idx] >= overlap and not true_used[idx]:
            TP[action_id] += 1
            true_used[idx] = 1
        else:
            FP[action_id] += 1

    TP = TP.sum()
    FP = FP.sum()
    # False negatives are any unused true segment (i.e. "miss")
    FN = n_true - true_used.sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)

    # If the precision + recall == 0, it is a NaN. Set these to 0.
    F1 = np.nan_to_num(F1)

    return F1


def overlap_f1_multiple_videos(unobserved_actions_per_video, predicted_actions_per_video,
                               action_to_id, num_classes=0, bg_class=None, overlap=0.1):
    overlap_f1s = []
    for unobserved_actions_per_frame, predicted_actions_per_frame in zip(unobserved_actions_per_video,
                                                                         predicted_actions_per_video):
        overlap_f1 = overlap_f1_single_video(unobserved_actions_per_frame,
                                             predicted_actions_per_frame,
                                             action_to_id=action_to_id,
                                             num_classes=num_classes, bg_class=bg_class, overlap=overlap)
        overlap_f1s.append(overlap_f1)
    return np.array(overlap_f1s).mean()


def segment_intervals(actions_per_frame):
    actions, lengths = aggregate_actions_and_lengths(actions_per_frame)
    actions_initial_frames = [0] + list(accumulate(lengths))
    return list(zip(actions_initial_frames[:-1], actions_initial_frames[1:]))


def do_transition_surgery(predicted_actions, la_length, transition_action, unobserved_actions):
    predicted_actions = predicted_actions[la_length:]
    if transition_action == unobserved_actions[0]:
        _, unobserved_actions_lengths = aggregate_actions_and_lengths(unobserved_actions)
        ta_length = unobserved_actions_lengths[0]
        predicted_actions = [transition_action] * ta_length + predicted_actions
    return predicted_actions


def do_future_surgery(predicted_actions, la_length, transition_action, unobserved_actions):
    predicted_actions = predicted_actions[:la_length]
    if transition_action == unobserved_actions[0]:
        ta_length = aggregate_actions_and_lengths(unobserved_actions)[1][0]
        predicted_actions += unobserved_actions[ta_length:]
    else:
        predicted_actions += unobserved_actions
    return predicted_actions


def extend_or_trim_predicted_actions(predicted_actions, unobserved_actions):
    """Extend or trim predicted actions to match the number of unobserved actions.

    If the list of predicted actions is smaller than the list of unobserved actions, the last action in the list
    of predicted actions is repeated until the length of the predicted actions match the length of the unobserved
    actions. If the list of predicted actions is longer than the list of unobserved actions, return the
    first predictions that match the length of the list of unobserved actions.

    Arg(s):
        predicted_actions - List containing the predicted actions.
        unobserved_actions - List containing the unobserved actions.
    Returns:
        The list of predicted actions either extended or trimmed to match the length of the unobserved actions.
    """
    if len(predicted_actions) < len(unobserved_actions):
        predicted_actions, _ = extend_smallest_list(predicted_actions, unobserved_actions)
    else:
        predicted_actions = predicted_actions[:len(unobserved_actions)]
    return predicted_actions


def extend_smallest_list(a, b, extension_val=None):
    """Extend the smallest list to match the length of the longest list.

    If extension_val is None, the extension is done by repeating the last element of the list. Otherwise, use
    extension_val.

    Arg(s):
        a - A list.
        b - A list.
        extension_val - Extension value.
    Returns:
        The input lists with the smallest list extended to match the size of the longest list.
    """
    gap = abs(len(a) - len(b))
    if len(a) > len(b):
        extension_val = extension_val if extension_val is not None else b[-1]
        b += [extension_val] * gap
    else:
        extension_val = extension_val if extension_val is not None else a[-1]
        a += [extension_val] * gap
    return a, b
