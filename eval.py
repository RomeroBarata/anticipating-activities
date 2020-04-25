#!/usr/bin/python2.7
import argparse
import glob
import os
import re

import numpy as np

from utils.analysis import analyse_observations_and_predictions, analyse_full_split_moc
from utils.analysis import analyse_performance_per_future_action, compute_segmental_edit_score_multiple_videos
from utils.analysis import overlap_f1_multiple_videos, do_transition_surgery, do_future_surgery
from utils.analysis import aggregate_actions_and_lengths, extend_or_trim_predicted_actions


def read_sequences(filename, ground_truth_path, obs_percentage, substitute_transition=False, substitute_future=False,
                   ignore_silence_action=None):
    # read ground truth
    gt_file = args.ground_truth_path + re.sub('\.recog', '.txt', re.sub('.*/', '/', filename))
    with open(gt_file, 'r') as f:
        ground_truth = f.read().split('\n')[0:-1]
        f.close()
    if ignore_silence_action is not None:
        ground_truth = [gt_action for gt_action in ground_truth if gt_action != ignore_silence_action]
    # read recognized sequence
    with open(filename, 'r') as f:
        recognized = f.read().split('\n')[1].split()
        f.close()
    
    last_frame = min(len(recognized), len(ground_truth))
    recognized = recognized[int(obs_percentage * len(ground_truth)):last_frame]
    obs_gt = ground_truth[:int(obs_percentage * len(ground_truth))]
    ta = obs_gt[-1]
    ground_truth = ground_truth[int(obs_percentage * len(ground_truth)):last_frame]

    if substitute_transition:
        la_length = aggregate_actions_and_lengths(ground_truth)[1][0]
        recognized = do_transition_surgery(recognized, la_length, ta, ground_truth)
        recognized = extend_or_trim_predicted_actions(recognized, ground_truth)
    elif substitute_future:
        la_length = aggregate_actions_and_lengths(ground_truth)[1][0]
        recognized = do_future_surgery(recognized, la_length, ta, ground_truth)
        recognized = extend_or_trim_predicted_actions(recognized, ground_truth)

    return ground_truth, recognized, obs_gt, ta


################################################################## 
parser = argparse.ArgumentParser()
parser.add_argument('--obs_perc')
parser.add_argument('--recog_dir')
parser.add_argument('--mapping_file', default='./data/mapping_bfc.txt')
parser.add_argument('--ground_truth_path', default='./data/groundTruth')
parser.add_argument('--ignore_silence_action', type=str)
parser.add_argument('--substitute_transition', action='store_true', help='TODO')
parser.add_argument('--substitute_future', action='store_true', help='TODO')
parser.add_argument('--do_error_analysis', action='store_true', help='TODO')

args = parser.parse_args()

    
obs_percentage = float(args.obs_perc)
classes_file = open(args.mapping_file, 'r')
content = classes_file.read()

classes = content.split('\n')[:-1]
action_to_id = {}
for i in range(len(classes)):
    action_id, action = classes[i].split()
    action_to_id[action] = int(action_id)
    classes[i] = classes[i].split()[1]
    
filelist = glob.glob(args.recog_dir + '/P*')
if not filelist:  # 50 Salads files do not start with P
    filelist = glob.glob(args.recog_dir + '/*.recog')

n_T = np.zeros(len(classes))
n_F = np.zeros(len(classes))

base_dir = os.path.dirname(args.recog_dir)
observe_str, pred_str = os.path.basename(args.recog_dir).split('-')[:2]
observe_str, pred_str = observe_str[-3:], pred_str[-3:]
save_dir = os.path.join(base_dir, observe_str + '_' + pred_str + '_analysis')
gts, recogs, transition_actions = [], [], []
for filename in filelist:
    gt, recog, obs, transition_action = read_sequences(filename, args.ground_truth_path, obs_percentage,
                                                       substitute_transition=args.substitute_transition,
                                                       substitute_future=args.substitute_future,
                                                       ignore_silence_action=args.ignore_silence_action)
    if args.do_error_analysis:
        analyse_observations_and_predictions(recog, obs, gt, action_to_id=action_to_id, save_path=save_dir,
                                             save_file_name=os.path.basename(filename))
    gts.append(gt)
    recogs.append(recog)
    transition_actions.append(transition_action)
    for i in range(len(gt)):
        if gt[i] == recog[i]:
            n_T[classes.index(gt[i])] += 1
        else:
            n_F[classes.index(gt[i])] += 1
##################################################################
acc = 0
n = 0
for i in range(len(classes)):
    if (n_T[i] + n_F[i]) != 0:
        acc += float(n_T[i]) / (n_T[i] + n_F[i])
        n += 1
try:
    moc = float(acc) / n
except ZeroDivisionError:
    moc = 0.0
print "MoC  %.4f" % moc

if args.do_error_analysis:
    analyse_full_split_moc(moc, n_T, n_F, action_to_id=action_to_id, save_path=save_dir)
    analyse_performance_per_future_action(recogs, gts, transition_actions, save_path=save_dir)
seg_edit_score = compute_segmental_edit_score_multiple_videos(gts, recogs)
for overlap in [0.10, 0.25, 0.50]:
    overlap_f1_score = overlap_f1_multiple_videos(gts, recogs, action_to_id=action_to_id,
                                                  num_classes=len(action_to_id), overlap=overlap)
    print "F1@%.2f: %.4f" % (overlap, overlap_f1_score)

gts, recogs = np.concatenate(gts), np.concatenate(recogs)
acc = np.mean(gts == recogs).item()
print "Frame-level Acc. %.4f" % acc
print "Segmental Edit Score: %.4f" % seg_edit_score
