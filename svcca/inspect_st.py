import cca_core

import torch
import torch.nn.functional as F

import torch.nn as nn
from scipy.spatial import distance
import numpy as np
import sys

# suffix of the activation tensors
suffix="activation.pt.norm"
# directory where the activations are saved
out_dir = sys.argv[1]

if len(sys.argv) == 2:
    sv=470
else:
    sv=int(sys.argv[2])

print("*** Results using SVCCA keeping {0} dims".format(sv))

from os import listdir
from os.path import isfile, join
f = [f for f in listdir(out_dir) if isfile(join(out_dir, f)) and f.endswith(suffix)]

print("*** Comparing pairwise between:")
print(f)

svcca = 0
cnt = 0
for i, f_i in enumerate(f):
    for j, f_j in enumerate(f):
        if i < j:
            act_nl = torch.load(out_dir + '/' + f_i, map_location=lambda storage, loc: storage)
            act_de = torch.load(out_dir + '/' + f_j, map_location=lambda storage, loc: storage)

            print(f_i, 'vs', f_j)

            for layer_idx in act_de:

                try:
                    assert len(act_nl[layer_idx]) == len(act_de[layer_idx])

                except AssertionError:
                    raise AssertionError("Minibatch size from the two sets of tensors must agree,"
                         "got {0:d} vs {1:d} instead".format(len(act_nl[layer_idx]), len(act_de[layer_idx])))

                all_batches_nl = []
                all_batches_de = []
                running_sum_cos = 0.0
                running_sum_pdist =0.0
                running_cnt = 0

                for batch_idx in range(len(act_nl[layer_idx])):
                    act_nl_meanpool = act_nl[layer_idx][batch_idx].sum(dim=0)
                    non_pad_len = (act_nl[layer_idx][batch_idx]!=0).bool().sum(dim=0)
                    act_nl_meanpool = act_nl_meanpool / non_pad_len

                    act_de_meanpool = act_de[layer_idx][batch_idx].sum(dim=0)
                    non_pad_len = (act_de[layer_idx][batch_idx]!=0).bool().sum(dim=0)
                    act_de_meanpool = act_de_meanpool / non_pad_len

                    all_batches_nl.append(act_nl_meanpool)
                    all_batches_de.append(act_de_meanpool)
            #       print(act_nl_meanpool.shape, act_de_meanpool.shape, act_nl_meanpool - act_de_meanpool)

                    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                    running_sum_cos += cos(act_de_meanpool, act_nl_meanpool).sum()

                    pdist = nn.PairwiseDistance(p=2, keepdim=True)
                    running_sum_pdist += (pdist(act_de_meanpool, act_nl_meanpool) / (torch.norm(act_de_meanpool[layer_idx]))).sum()

                    running_cnt += act_de_meanpool.shape[0]

                all_nl = torch.cat(all_batches_nl, 0).t().numpy()
                all_de = torch.cat(all_batches_de, 0).t().numpy()

                # MSE and cosine distance
            #     results = cca_core.get_cca_similarity(all_de, all_nl, verbose=False)
            #     print("Similarity without SVD:{:.4f}".format(np.mean(results["cca_coef1"])))

                # Mean subtract activations
                cacts1 = all_de - np.mean(all_de, axis=1, keepdims=True)
                cacts2 = all_nl - np.mean(all_nl, axis=1, keepdims=True)

                # Perform SVD
                U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
                U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

                svacts1 = np.dot(s1[:sv]*np.eye(sv), V1[:sv])
                svacts2 = np.dot(s2[:sv]*np.eye(sv), V2[:sv])

                svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)

                print("Similarity after SVD:{:.4f}".format(np.mean(svcca_results["cca_coef1"])))
                print("==============================================")

                svcca += np.mean(svcca_results["cca_coef1"])
                cnt += 1

print("Avg. SVCCA over {0} pairs: {1}".format(cnt, svcca / cnt))

print("Done.")

