import argparse
import glob
import os
import re


parser = argparse.ArgumentParser()
parser.add_argument('-model_dir', required=True,
                    help='Path to the directory that stores the models')

opt = parser.parse_args()
model_paths = glob.glob(opt.model_dir + "/*.pt", recursive=False)
best_model_name = os.path.basename(model_paths[0])
best_ppl = float(re.search('model_ppl_(.+?)_', best_model_name).group(1))
for model_path in model_paths:
    model_name = os.path.basename(model_path)
    ppl = float(re.search('model_ppl_(.+?)_', model_name).group(1))
    if ppl < best_ppl:
        best_model_name = model_name
        best_ppl = ppl
print(best_model_name)