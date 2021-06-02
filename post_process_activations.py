import sys
from collections import defaultdict
import torch
import glob
import os
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('-activations_dir', required=True,
                    help='Path to the directory where the activations are saved.')
parser.add_argument('-save_activation', required=True,
                    help='Location to save the appended activations.')

""" Append the final encoder outputs of mini-batches """

opt = parser.parse_args()

# Get list of file names in datetime str format and sort them
file_names = [os.path.split(x)[1].replace('.norm', '') for x in glob.glob(f"{opt.activations_dir}/*.norm")]
file_names.sort(key=lambda date: datetime.strptime(date, '%Y-%m-%d-%H:%M:%S.%f'))

# Init dictionary where key=layer_idx, value=list of activations where each element is a minibatch of activations
saved_att = defaultdict(list)

for file_name in file_names:
    padded_context = torch.load(f"{opt.activations_dir}/{file_name}.norm")
    # key=-1 to symbolize last encoder output (after layer norm)
    saved_att[-1].append(padded_context)

torch.save(saved_att, opt.save_activation + '.norm')