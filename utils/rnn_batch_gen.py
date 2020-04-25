#!/usr/bin/python2.7
import random

import numpy as np

from analysis import aggregate_actions_and_lengths
from base_batch_gen import Base_batch_generator
from helper_functions import get_label_length_seq, split_multi_label_seq


class RNN_batch_generator(Base_batch_generator):
    def __init__(self, nClasses, n_iterations, max_seq_sz, actions_dict, alpha, num_parent_classes=0,
                 parent_actions_dict=None):
        super(RNN_batch_generator, self).__init__()
        self.n_iterations = n_iterations
        self.num_parent_classes = num_parent_classes
        self.nClasses = nClasses
        self.max_seq_sz = max_seq_sz
        self.parent_actions_dict = parent_actions_dict
        self.actions_dict = actions_dict
        self.alpha = alpha

    def read_data(self, list_of_videos, list_of_fisher_vectors=None, ignore_silence_action=None,
                  list_of_parent_videos=None):
        for vid_index, vid in enumerate(list_of_videos):
            file_ptr = open(vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            if ignore_silence_action is not None:
                content = [action for action in content if action != ignore_silence_action]
            
            label_seq, length_seq = get_label_length_seq(content) 
            T = (1.0 / self.alpha) * len(content)

            # TODO: Check this is working correctly
            parent_label_seq = None
            if list_of_parent_videos is not None:
                file_ptr = open(list_of_parent_videos[vid_index], 'r')
                parent_content = file_ptr.read().split('\n')[:-1]
                if ignore_silence_action is not None:
                    parent_content = [action for action in parent_content if action != ignore_silence_action]
                multi_label_seq, _ = aggregate_actions_and_lengths(list(zip(parent_content, content)))
                parent_label_seq, _ = split_multi_label_seq(multi_label_seq)

            for itr in range(self.n_iterations):
                # list of partial length of each label in the sequence
                rand_cuts = []
                for i in range(len(label_seq) - 1):
                    cut = int(length_seq[i] * float(itr + .5) / self.n_iterations)
                    rand_cuts.append(cut)

                for i in range(len(rand_cuts)):
                    seq_len = i + 1
                    p_seq = []
                    for j in range(seq_len):
                        p_seq.append(np.zeros((self.num_parent_classes + self.nClasses + 1)))
                        if j == seq_len - 1:
                            p_seq[-1][-1] = rand_cuts[j] / T
                        else:
                            p_seq[-1][-1] = length_seq[j] / T
                        p_seq[-1][self.num_parent_classes + self.actions_dict[label_seq[j]]] = 1
                        if parent_label_seq is not None:
                            p_seq[-1][self.parent_actions_dict[parent_label_seq[j]]] = 1

                    for j in range(self.max_seq_sz - seq_len):
                        p_seq.append(np.zeros((self.num_parent_classes + self.nClasses + 1)))
                    
                    p_tar = np.zeros((self.num_parent_classes + self.nClasses + 2))
                    # target length
                    if i != len(rand_cuts) - 1:
                        p_tar[-2] = rand_cuts[i + 1] / T
                    else:
                        p_tar[-2] = length_seq[i + 1] / T
                    # remaining length
                    p_tar[-1] = (length_seq[i] - rand_cuts[i]) / T
                    # target action
                    p_tar[ self.num_parent_classes + self.actions_dict[label_seq[i + 1]] ] = 1
                    if parent_label_seq is not None:
                        p_tar[self.parent_actions_dict[parent_label_seq[i + 1]]] = 1
                    
                    example = [p_seq, p_tar, seq_len]
                    self.list_of_examples.append(example)
        random.shuffle(self.list_of_examples) 
        return

    def next_batch(self, batch_size):
        batch_data = sorted(self.list_of_examples[self.index:self.index+batch_size], key=lambda x: x[2], reverse=True)
        batch = np.array(batch_data)
        self.index += batch_size
        batch_vid = list(batch[:, 0])
        batch_target = list(batch[:, 1])
                
        return batch_vid, batch_target
