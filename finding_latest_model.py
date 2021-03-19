import argparse
import glob
import os
import re


parser = argparse.ArgumentParser()
parser.add_argument('-model_dir', required=True,
                    help='Path to the directory that stores the models')

opt = parser.parse_args()
model_paths = glob.glob(opt.model_dir + "/*.pt", recursive=False)
latest_model_name = os.path.basename(model_paths[0])
latest_epoch = int(re.search(r'_e(.*?)\.00\.pt', latest_model_name).group(1))
for model_path in model_paths:
    model_name = os.path.basename(model_path)
    epoch = int(re.search(r'_e(.*?)\.00\.pt', model_name).group(1))
    if epoch > latest_epoch:
        latest_model_name = model_name
        latest_epoch = epoch
print(latest_model_name)