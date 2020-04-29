#!/usr/bin/python2.7
import os

import numpy as np

from analysis import aggregate_actions_and_lengths


def read_mapping_dict(mapping_file):
    """
    read a mapping dictionary between the action labels and their IDs
    """
    file_ptr = open(mapping_file, 'r') 
    actions = file_ptr.read().split('\n')[:-1]
    
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
        
    return actions_dict


def encode_content(content, nRows, nCols, actions_dict):
    """
    Encode a sequence of actions into a matrix form for the cnn model
    """
    encoded_content = np.zeros([nRows, nCols])

    start = 0
    s = 0
    e = 0
    for i in range(len(content)):
        if content[i] != content[start]:
            frame_label = np.zeros((nCols))
            frame_label[ actions_dict[content[start]] ] = 1
            s = int( nRows*(1.0*start/len(content)) )
            e = int( nRows*(1.0*i/len(content)) )
            encoded_content[s:e] = frame_label
            start = i
    frame_label = np.zeros((nCols))
    frame_label[ actions_dict[content[start]] ] = 1
    encoded_content[e:] = frame_label

    return encoded_content


def write_predictions(path, f_name, recognition):
    """
    Write the prediction output to a file
    """
    if not os.path.exists(path):
        os.makedirs(path)
    f_ptr = open(path + "/" + f_name + ".recog", "w")

    f_ptr.write("### Frame level recognition: ###\n")
    f_ptr.write(' '.join(recognition))
    
    f_ptr.close()


def write_step_predictions(path, f_name, obs_label_seq, obs_length_seq, unobs_label_seq, unobs_length_seq,
                           pred_label_seq, pred_length_seq):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, f_name + '.txt'), mode='w') as f:
        f.write('Observed\n')
        for obs_label, obs_length in zip(obs_label_seq, obs_length_seq):
            f.write('\t' + obs_label.ljust(24, ' ') + '\t' + str(obs_length).rjust(4, ' ') + '\n')
        f.write('\nUnobserved\n')
        for unobs_label, unobs_length in zip(unobs_label_seq, unobs_length_seq):
            f.write('\t' + unobs_label.ljust(24, ' ') + '\t' + str(unobs_length).rjust(4, ' ') + '\n')
        f.write('\nPredicted\n')
        transition_obs_length, is_first = obs_length_seq[-1], True
        for pred_label, pred_length in zip(pred_label_seq, pred_length_seq):
            if is_first:
                f.write('\t' + pred_label.ljust(24, ' ') + '\t' +
                        str(int(round(pred_length - transition_obs_length))).rjust(4, ' ') + '\n')
                is_first = False
            else:
                f.write('\t' + pred_label.ljust(24, ' ') + '\t' + str(int(round(pred_length))).rjust(4, ' ') + '\n')


def get_label_length_seq(content):
    """
    Get the sequence of labels and length for a given frame-wise action labels
    """
    label_seq = []
    length_seq = []
    start = 0
    for i in range(len(content)):
        if content[i] != content[start]:
            label_seq.append(content[start])
            length_seq.append(i-start)
            start = i
    label_seq.append(content[start])
    length_seq.append(len(content)-start)
    
    return label_seq, length_seq


def split_multi_label_seq(multi_label_seq):
    parent_seq, child_seq = [], []
    for parent_action, child_action in multi_label_seq:
        parent_seq.append(parent_action)
        child_seq.append(child_action)
    return parent_seq, child_seq


def get_parent_label_seq(parent_content, child_content):
    if parent_content is None:
        return parent_content
    multi_label_seq, _ = aggregate_actions_and_lengths(list(zip(parent_content, child_content)))
    parent_seq, _ = split_multi_label_seq(multi_label_seq)
    return parent_seq


def filter_lists(list_of_videos, list_of_fisher_vectors):
    videos_prefix = os.path.dirname(list_of_videos[0])
    fisher_prefix = os.path.dirname(list_of_fisher_vectors[0])

    video_files = {os.path.basename(video) for video in list_of_videos}
    fisher_files = {os.path.basename(fisher_vector) for fisher_vector in list_of_fisher_vectors}
    files = sorted(video_files & fisher_files)
    list_of_videos = [os.path.join(videos_prefix, file) for file in files]
    list_of_fisher_vectors = [os.path.join(fisher_prefix, file) for file in files]
    return list_of_videos, list_of_fisher_vectors
