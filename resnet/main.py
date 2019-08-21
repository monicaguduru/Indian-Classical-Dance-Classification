import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn
import csv
from tqdm import tqdm

from data import DataSet
from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video


if __name__=="__main__":
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 16
    opt.n_classes = 6

    model = generate_model(opt)
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    # print(model_data['arch'])
    # print("11111111111111111111111111111111111111111111")
    # print(opt.arch)
    assert opt.arch == model_data['arch']
    # print(model_data['state_dict'])
    # model.load_state_dict(model_data['state_dict'])
    model.eval()
    if opt.verbose:
        print(model)
    # Set defaults.
    seq_length = 30
    class_limit = 6# Number of classes to extract. Can be 1-101 or None for all.

    # Get the dataset.
    data = DataSet(seq_length=seq_length, class_limit=class_limit)

    input_files = []
    # with open(opt.input, 'r') as f:
    #     for row in f:
    #         input_files.append(row[:-1])
    pbar = tqdm(total=len(data.data))
    # print("--------------------------------------------------")
    # print(data.data)
    # print(input_files)
    class_names = []
    with open('class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])

    for video in data.data:
        # Get the path to the sequence for this video.
        path = os.path.join('../btp','data1', 'sequences', video[1]+video[2] + '-' + str(seq_length) + \
            '-features')  # numpy will auto-append .npy
        if (video[0]=='train'):
            dir_name = os.path.join('../btp','data','train', video[1], video[2])
        elif(video[0]=='test'):
            dir_name = os.path.join('../btp','data','test', video[1], video[2])

        # print(dir_name)
        # Check if we already have it.
        if os.path.isfile(path + '.npy'):
            pbar.update(1)
            continue

        # # Get the frames for this video.
        # frames = data.get_frames_for_sample(video)

        # # Now downsample to just the ones we need.
        # frames = data.rescale_list(frames, seq_length)

        sequence = []
        input_file = video[2]
        # for image in frames:
        # features = model.extract(image)
        # sequence.append(features)
        print(dir_name)

        features = classify_video(dir_name, input_file, class_names, model, opt)

        np.save(path, sequence)
        pbar.update(1)

    pbar.close()


    # ffmpeg_loglevel = 'quiet'
    # if opt.verbose:
    #     ffmpeg_loglevel = 'info'

    # if os.path.exists('tmp'):
    #     subprocess.call('rm -rf tmp', shell=True)

    # outputs = []
    # for input_file in input_files:
    #     video_path = os.path.join(opt.video_path, input_file)
    #     print(video_path)
    #     if os.path.exists(video_path):
    #         print(video_path)
    #         # subprocess.call('mkdir tmp', shell=True)
    #         # subprocess.call('ffmpeg -i {} tmp/image_%05d.jpg'.format(video_path),
    #         #                 shell=True)

    #         result = classify_video('tmp', input_file, class_names, model, opt)
    #         outputs.append(result)
    #         clips.features

    #         # subprocess.call('rm -rf tmp', shell=True)
    #     else:
    #         print('{} does not exist'.format(input_file))

    # if os.path.exists('tmp'):
    #     subprocess.call('rm -rf tmp', shell=True)

    # with open(opt.output, 'w') as f:
    #     json.dump(outputs, f)