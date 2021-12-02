cuda_id = 0
limit_N = 16
limit_k = 1.5

test_seqs_file = 'dataset/motif.pkl'
dataset = 'DEEPre'
# dataset = 'ECPred'
class_head = 'main'
# class_head = 'sub1'
# class_head = 'lv0'
# class_head = 'lv1'

model_name = f"{dataset}_{class_head}_ly6_test.pkl"
test_ids = []

import torch
from utils.env import *
from model.xfmr_att_fc import DAttProt_cls, feat_agreement
import pandas as pd
from tqdm import tqdm

device = torch.device(f"cuda:{cuda_id}")
model = torch.load(os.path.join(model_root, model_name), map_location=f"cuda:{cuda_id}")

net = DAttProt_cls(**model['args']).to(device)
net.load_state_dict(model['model'])
net.eval()


def get_att(x):
    # embed: (1, L, F)
    # mask: (1, L)
    embed, mask = net.feat_embedding(x)
    # (1, L, F)
    feats = net.coder(embed, mask, net.trn_ly)

    # (1, L, F_out)
    feats_local = net.norm(net.dim_reduction(feats))
    # [N * (1, L, F_out)]
    feats_multi = []
    for conv in net.convs:
        # (1, F_out, L)
        feat = conv(feats.transpose(-2, -1).contiguous())[:, :, :net.l]
        # (1, L, F_out)
        feats_multi.append(feat.transpose(-2, -1).contiguous())
    # (1, N, L, F_out)
    feats_multi = net.norm(torch.stack(feats_multi, dim=1))
    # (1, 1, L, F_out)
    feats_center = (feats_multi.sum(dim=1) + feats_local).unsqueeze(1) / (len(net.scales) + 1)

    # feats: (1, L, F_out)
    # pos_probs: (1, 1, L)
    # scale_probs: (1, N, L)
    _, pos_probs, scale_probs = feat_agreement(feats_multi, mask, feats_center)
    return pos_probs.squeeze(0), scale_probs.squeeze(0)


data = {}
test_seqs = pd.read_pickle(test_seqs_file).to_dict()
seq_len = model['args']['seq_len']
scales = model['args']['scales']

for seq in tqdm(test_seqs, desc="prepare sequences"):
    if dataset is not None and dataset not in test_seqs[seq]['dataset']:
        continue
    if test_ids and test_seqs[seq]['id'] not in test_ids:
        continue
    # if len(test_seqs[seq]['feats']) <= 2:
    #     continue
    data[seq] = {
        'tensor': torch.zeros((1, seq_len), dtype=torch.long).to(device),
        'feats': test_seqs[seq]['feats'],
        'id': test_seqs[seq]['id']
    }
    for i, c in enumerate(seq):
        if i >= seq_len:
            break
        data[seq]['tensor'][0, i] = token2ix[c]

motifs_record = {}

for seq in tqdm(data, desc="processing sequences"):
    # pos_p: (1, L)
    # scale_p: (N, L)
    with torch.no_grad():
        pos_p, scale_p = get_att(data[seq]['tensor'])
        double_p = pos_p * scale_p
        id = data[seq]['id']
        feats = data[seq]['feats']
        motifs_truth = []
        for feat in feats:
            start, end = feats[feat]['left'] - 1, feats[feat]['right'] - 1
            mid = (start + end) // 2
            size = end - start + 1
            motifs_truth.append((mid, size, seq[start:end + 1], feat, start, end))
        # (N_limit,)
        key_pos_p, key_pos = torch.topk(pos_p.squeeze(), limit_N)
        key_pos_p_scale = scale_p[:, key_pos]
        key_pos_key_scale_p, key_pos_key_scale = torch.max(key_pos_p_scale, dim=0)
        motifs_pred = []

        for pos, p_pos, scale in zip(key_pos, key_pos_p, key_pos_key_scale):
            if p_pos < limit_k / len(seq):
                continue
            size = scales[scale]
            start = max(0, pos - size // 2)
            end = min(seq_len - 1, pos + size // 2)
            motifs_pred.append((pos.item(), size, seq[start:end + 1]))

        motifs_truth.sort(key=lambda x: x[0])
        motifs_pred.sort(key=lambda x: x[0])
        motifs_record[id] = {'seq': seq, 'truth': motifs_truth, 'pred': motifs_pred}

for id in motifs_record:
    seq = motifs_record[id]
    t = motifs_record[id]['truth']
    p = motifs_record[id]['pred']
    if p and p[-1][0] >= t[0][0] and p[0][0] <= t[-1][0]:
        i = j = 0
        while i < len(p) and j < len(t):
            if p[i][0] == t[j][0]:
                diff = abs(p[i][1] - t[j][1])
                diff_ = min([abs(t[j][1] - s) for s in scales])
                if diff == diff_:
                    print(f"{id}#{dataset}#{t[j][3]}#{t[j][4]}#{t[j][5]}")
                i += 1
                j += 1
            elif p[i][0] < t[j][0]:
                i += 1
            else:
                j += 1
pd.to_pickle(motifs_record, os.path.join(model_root, f"interpretability of {model_name}"))
