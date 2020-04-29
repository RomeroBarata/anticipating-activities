#!/usr/bin/python2.7
import argparse

import numpy as np
import tensorflow as tf

from models.cnn import ModelCNN
from models.rnn import ModelRNN
from utils.base_batch_gen import Base_batch_generator
from utils.cnn_batch_gen import CNN_batch_generator
from utils.rnn_batch_gen import RNN_batch_generator
from utils.cnn_fisher_batch_gen import CNNFisherBatchGen
from utils.helper_functions import read_mapping_dict, encode_content, write_predictions, get_label_length_seq
from utils.helper_functions import filter_lists, write_step_predictions, get_parent_label_seq

parser = argparse.ArgumentParser()

parser.add_argument("--model", default="rnn", help="select model: [\"rnn\", \"cnn\"]")
parser.add_argument("--action", default="predict", help="select action: [\"train\", \"predict\"]")

parser.add_argument("--mapping_file", default="./data/mapping_bf.txt")
parser.add_argument("--vid_list_file", default="./data/test.split1.bundle")
parser.add_argument('--mapping_parent_file', type=str)
parser.add_argument('--parent_vid_list_file', type=str)
parser.add_argument('--fisher_list_file', default=None, type=str)
parser.add_argument('--ignore_silence_action', type=str)
parser.add_argument("--model_save_path", default="./save_dir/models/rnn")
parser.add_argument("--results_save_path", default="./save_dir/results/rnn")

parser.add_argument("--save_freq", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--nEpochs", type=int, default=20)
parser.add_argument("--eval_epoch", type=int, default=20)

# RNN specific parameters
parser.add_argument("--rnn_size", type=int, default=256)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--max_seq_sz", type=int, default=25)
parser.add_argument("--alpha", type=float, default=6,
                    help="a scalar value used in normalizing the input length. It is the average number of actions "
                         "in the videos.")
parser.add_argument("--n_iterations", type=int, default=10,
                    help="number of training examples corresponding to each action segment for the rnn")

# CNN specific parameters
parser.add_argument("--nRows", type=int, default=128)
parser.add_argument("--sigma", type=int, default=3, help="sigma for the gaussian smoothing step")

# Test on GT or decoded input
parser.add_argument("--input_type", default="gt", help="select input type: [\"decoded\", \"gt\", \"fisher\"]")
parser.add_argument("--decoded_path", default="./data/decoded/split1")

########################################################################################################################

args, unknown = parser.parse_known_args()
ignore_silence_action = args.ignore_silence_action

actions_dict = read_mapping_dict(args.mapping_file)
nClasses = len(actions_dict)

file_ptr = open(args.vid_list_file, 'r') 
list_of_videos = file_ptr.read().split('\n')[1:-1]
list_of_fisher_vectors = None
if args.fisher_list_file is not None:
    with open(args.fisher_list_file, mode='r') as f:
        list_of_fisher_vectors = f.read().split('\n')[1:-1]
    list_of_videos, list_of_fisher_vectors = filter_lists(list_of_videos, list_of_fisher_vectors)
parent_actions_dict, list_of_parent_videos = None, None
parent_vid_list_file = args.parent_vid_list_file
if parent_vid_list_file is not None:
    with open(parent_vid_list_file, mode='r') as f:
        list_of_parent_videos = f.read().split('\n')[1:-1]
    parent_actions_dict = read_mapping_dict(args.mapping_parent_file)
num_parent_classes = len(parent_actions_dict) if parent_actions_dict is not None else 0

################
# Training #####
################
if args.action == "train":
    model = None
    batch_gen = Base_batch_generator()
    
    if args.model == "rnn":
        model = ModelRNN(nClasses, args.rnn_size, args.max_seq_sz, args.num_layers,
                         num_parent_classes=num_parent_classes)
        batch_gen = RNN_batch_generator(nClasses, args.n_iterations, args.max_seq_sz, actions_dict, args.alpha,
                                        num_parent_classes=num_parent_classes, parent_actions_dict=parent_actions_dict)
    elif args.model == "cnn":
        if args.fisher_list_file is not None:
            model = ModelCNN(nRows=args.nRows, nCols=64, nClasses=nClasses)
            batch_gen = CNNFisherBatchGen(num_rows=args.nRows, num_classes=nClasses, action_to_id=actions_dict)
        else:
            model = ModelCNN(nRows=args.nRows, nCols=nClasses)
            batch_gen = CNN_batch_generator(args.nRows, nClasses, actions_dict)
        
    batch_gen.read_data(list_of_videos, list_of_fisher_vectors, ignore_silence_action=ignore_silence_action,
                        list_of_parent_videos=list_of_parent_videos)
    with tf.Session() as sess:
        model.train(sess, args.model_save_path, batch_gen, args.nEpochs, args.save_freq, args.batch_size)

##################
# Prediction #####
##################
elif args.action == "predict":
    obs_percentages = [.2, .3]
    model_restore_path = args.model_save_path + "/epoch-" + str(args.eval_epoch) + "/model.ckpt"
    if "split1" in args.vid_list_file:
        data_split = '01'
    elif 'split2' in args.vid_list_file:
        data_split = '02'
    elif 'split3' in args.vid_list_file:
        data_split = '03'
    else:
        data_split = '04'
    
    if args.model == "rnn":
        model = ModelRNN(nClasses, args.rnn_size, args.max_seq_sz, args.num_layers,
                         num_parent_classes=num_parent_classes)
        for vid_index, vid in enumerate(list_of_videos):
            f_name = vid.split('/')[-1].split('.')[0]
            observed_content = []
            vid_len = 0
            if args.input_type == "gt":
                file_ptr = open(vid, 'r') 
                content = file_ptr.read().split('\n')[:-1]  # actions_per_frame
                if ignore_silence_action is not None:
                    content = [action for action in content if action != ignore_silence_action]
                vid_len = len(content)  # num_frames (whole video)
            parent_content = None
            if list_of_parent_videos is not None:
                file_ptr = open(list_of_parent_videos[vid_index], mode='r')
                parent_content = file_ptr.read().split('\n')[:-1]  # parent_actions_per_frame
                if ignore_silence_action:
                    parent_content = [action for action in parent_content if action != ignore_silence_action]
                
            for obs_p in obs_percentages:
                observed_parent_content = None
                unobserved_parent_content = None
                if args.input_type == "decoded":
                    # file_ptr = open(args.decoded_path + "/obs" + str(obs_p) + "/" + f_name + '.txt', 'r')
                    file_ptr = open(args.decoded_path + "/obs" + str(obs_p) + "/" + "S" + data_split + "/" + f_name + '.txt', 'r')
                    observed_content = file_ptr.read().split('\n')[:-1]
                    vid_len = int(len(observed_content) / obs_p)
                elif args.input_type == "gt":
                    observed_content = content[:int(obs_p * vid_len)]  # observed_actions_per_frame
                    try:
                        observed_parent_content = parent_content[:int(obs_p * vid_len)]
                        unobserved_parent_content = parent_content[int(obs_p * vid_len):]
                    except TypeError:
                        pass
                unobserved_content = content[int(obs_p * vid_len):]
                T = (1.0 / args.alpha) * vid_len
                
                pred_percentages = [.1, .2, .3, .5, .7, .8]
                pred_percentages = [pred_p for pred_p in pred_percentages if obs_p + pred_p <= 1.0]
                for pred_p in pred_percentages:
                    pred_len = int(pred_p * vid_len)  # num_frames_to_predict
                    output_len = pred_len + len(observed_content)  # write out observed + predictions
                    
                    label_seq, length_seq = get_label_length_seq(observed_content)
                    parent_label_seq = get_parent_label_seq(observed_parent_content, observed_content)
                    # We need copy of these lists because model.predict modifies them inplace.
                    obs_parent_label_seq = list(parent_label_seq)
                    obs_label_seq, obs_length_seq = list(label_seq), list(length_seq)
                    num_obs_actions = len(obs_label_seq)
                    with tf.Session() as sess:
                        label_seq, length_seq, parent_label_seq = model.predict(sess, model_restore_path, pred_len,
                                                                                label_seq, length_seq, actions_dict, T,
                                                                                parent_label_seq=parent_label_seq,
                                                                                parent_actions_dict=parent_actions_dict)
                    recognition, parent_recognition = [], []
                    for i in range(len(label_seq)):
                        recognition = np.concatenate((recognition, [label_seq[i]] * int(length_seq[i])))
                        if parent_label_seq is not None:
                            parent_recognition = np.concatenate((parent_recognition,
                                                                 [parent_label_seq[i]] * int(length_seq[i])))
                    recognition = recognition[:output_len]
                    parent_recognition = parent_recognition[:output_len]
                    # write results to file
                    f_name = vid.split('/')[-1].split('.')[0]
                    path = args.results_save_path + "/obs" + str(obs_p) + "-pred" + str(pred_p)
                    if args.input_type == "decoded":
                        path += "-noisy"
                    write_predictions(path, f_name, recognition)
                    if not isinstance(parent_recognition, list):
                        write_predictions(path + '_parent', f_name, parent_recognition)
                    # Stepwise output. If the model had both coarse and fine as input and output, the stepwise output
                    # will look like add_milk/grab_milk ...
                    unobs_label_seq, unobs_length_seq = get_label_length_seq(unobserved_content[:pred_len])
                    pred_label_seq = label_seq[num_obs_actions - 1:]
                    pred_length_seq = length_seq[:num_obs_actions - 1:]
                    if not isinstance(parent_recognition, list):
                        obs_label_seq = [pl + '/' + cl for pl, cl in zip(obs_parent_label_seq, obs_label_seq)]
                        unobs_parent_label_seq = get_parent_label_seq(unobserved_parent_content, unobserved_content)
                        unobs_label_seq = [pl + '/' + cl for pl, cl in zip(unobs_parent_label_seq, unobs_label_seq)]
                        pred_label_seq = [pl + '/' + cl
                                          for pl, cl in zip(parent_label_seq[num_obs_actions - 1:],
                                                            label_seq[num_obs_actions - 1:])]
                    write_step_predictions(path + '_stepwise', f_name, obs_label_seq, obs_length_seq, unobs_label_seq,
                                           unobs_length_seq, pred_label_seq, pred_length_seq)

    elif args.model == "cnn":
        if args.fisher_list_file is not None:
            model = ModelCNN(nRows=args.nRows, nCols=64, nClasses=nClasses)
        else:
            model = ModelCNN(args.nRows, nClasses)
        for vid_index, vid in enumerate(list_of_videos):
            f_name = vid.split('/')[-1].split('.')[0]
            observed_content = []
            vid_len = 0
            if args.input_type == "gt":
                file_ptr = open(vid, 'r') 
                content = file_ptr.read().split('\n')[:-1]
                vid_len = len(content)
            elif args.input_type == "fisher":
                fisher_content = np.loadtxt(list_of_fisher_vectors[vid_index], dtype=np.float32, ndmin=2)
                fisher_content = fisher_content[:, 1:]  # remove frame index
                file_ptr = open(vid, 'r')
                content = file_ptr.read().split('\n')[:-1]
                vid_len = len(content)

            for obs_p in obs_percentages:
                if args.input_type == "decoded":
                    file_ptr = open(args.decoded_path + "/obs" + str(obs_p) + "/" + f_name + '.txt', 'r')
                    observed_content = file_ptr.read().split('\n')[:-1]
                    vid_len = int(len(observed_content) / obs_p)
                elif args.input_type == "gt":
                    observed_content = content[:int(obs_p * vid_len)]
                elif args.input_type == "fisher":
                    observed_fisher_content = fisher_content[:int(obs_p * vid_len)]
                    observed_content = content[:int(obs_p * vid_len)]

                if args.input_type == "fisher":
                    input_x = [CNNFisherBatchGen.up_or_down_sample_fisher_vector(observed_fisher_content)]
                else:
                    input_x = encode_content(observed_content, args.nRows, nClasses, actions_dict)
                    input_x = [np.reshape(input_x, [args.nRows, nClasses, 1])]
                
                with tf.Session() as sess:
                    label_seq, length_seq = model.predict(sess, model_restore_path, input_x, args.sigma, actions_dict)
                    
                recognition = []
                for i in range(len(label_seq)):
                    recognition = np.concatenate((recognition,
                                                  [label_seq[i]] * int(0.5 * vid_len * length_seq[i] / args.nRows)))
                recognition = np.concatenate((observed_content, recognition))
                diff = int((0.5 + obs_p) * vid_len) - len(recognition)
                for i in range(diff):
                    recognition = np.concatenate((recognition, [label_seq[-1]]))
                # write results to file
                pred_percentages = [.1, .2, .3, .5, .7, .8]
                pred_percentages = [pred_p for pred_p in pred_percentages if obs_p + pred_p <= 1.0]
                for pred_p in pred_percentages:
                    path = args.results_save_path + "/obs" + str(obs_p) + "-pred" + str(pred_p)
                    write_predictions(path, f_name, recognition[:int((pred_p + obs_p) * vid_len)])
