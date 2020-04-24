#!/usr/bin/python2.7
import random

import numpy as np
import torch
import torch.nn.functional as F

from base_batch_gen import Base_batch_generator
from helper_functions import encode_content


class CNNFisherBatchGen(Base_batch_generator):
    def __init__(self, num_rows, num_classes, action_to_id):
        super(CNNFisherBatchGen, self).__init__()
        self.num_rows = num_rows
        self.num_classes = num_classes
        self.action_to_id = action_to_id

    def read_data(self, list_of_videos, list_of_fisher_vectors=None):
        for video, fisher_file in zip(list_of_videos, list_of_fisher_vectors):
            fisher_vectors = np.loadtxt(fisher_file, dtype=np.float32, ndmin=2)
            fisher_vectors = fisher_vectors[:, 1:]  # remove frame index

            with open(video, mode='r') as f:
                actions_per_frame = [line.rstrip() for line in f]
            num_frames = len(actions_per_frame)

            observed_fractions = [0.1, 0.2, 0.3, 0.5]
            for observed_fraction in observed_fractions:
                num_observed_frames = int(observed_fraction * num_frames)
                num_observed_plus_unobserved_frames = int((0.5 + observed_fraction) * num_frames)

                observed_fisher_vector = fisher_vectors[:num_observed_frames]
                input_video = self.up_or_down_sample_fisher_vector(observed_fisher_vector)

                target = actions_per_frame[num_observed_frames:num_observed_plus_unobserved_frames]
                target = encode_content(target, self.num_rows, self.num_classes, self.action_to_id)
                target = np.reshape(target, [self.num_rows, self.num_classes, 1])
                example = [input_video, target]
                self.list_of_examples.append(example)
            random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size
        batch_input = []
        batch_target = []
        for batch_example in batch:
            batch_input.append(batch_example[0])
            batch_target.append(batch_example[1])
        return batch_input, batch_target

    @staticmethod
    def up_or_down_sample_fisher_vector(observed_fisher_vector):
        observed_fisher_vector = torch.from_numpy(observed_fisher_vector).permute([1, 0])
        observed_fisher_vector = torch.unsqueeze(observed_fisher_vector, dim=0)
        observed_fisher_vector = F.interpolate(observed_fisher_vector, size=128, mode='linear')
        observed_fisher_vector = observed_fisher_vector.squeeze().permute([1, 0])
        observed_fisher_vector = torch.unsqueeze(observed_fisher_vector, dim=-1).numpy()
        return observed_fisher_vector
