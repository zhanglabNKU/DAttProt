import re
import pandas as pd
from utils.timer import Timer
swissprot = 'dataset/uniprot_sprot.dat'
deepre = '../../dataset/DEEPre/DEEPre_data.pkl'
ecpred = '../../dataset/ECPred/ECPred_data.pkl'
motif_pkl = 'dataset/motif.pkl'
motif_json = 'dataset/motif.json'
rec = {}
id = ''
seq = ''
feats = {}
start = end = -1


def str2num(s):
    try:
        s = int(s)
        return s
    except ValueError:
        i = -1
        while s[i] in '0123456789':
            i -= 1
        if i < -1:
            return int(s[i + 1:])
        i = 0
        while s[i] in '0123456789':
            i += 1
        if i > 0:
            return int(s[:i])
        return -1


timer = Timer(True)

with open(swissprot) as f:
    line = 'START'
    while line:
        line = f.readline()
        mark = line[:2]
        if mark == 'ID':
            id = line.split()[1]
        if mark == 'FT':
            if 'MOTIF' in line:
                pos = line[20:-1]
                if '..' not in pos:
                    start = end = str2num(pos)
                else:
                    pos = pos.split('..')
                    start = str2num(pos[-2])
                    end = str2num(pos[-1])
                if start - end > 25:
                    start = end = -1
            if 'note=' in line:
                t = re.search(r".*note=\"(.*?)\".*", line)
                while t is None:
                    line = line[:-1] + f.readline()[20:]
                    t = re.search(r".*note=\"(.*?)\".*", line)
                if start >= 0:
                    feats[t.group(1)] = {'left': start, 'right': end}
                    start = end = -1
        if mark == 'SQ':
            line = f.readline()
            while line and line[:2] != '//':
                seq += line.replace(' ', '')[:-1]
                line = f.readline()
            if seq not in rec and 0 < len(seq) <= 512 and len(feats) > 0:
                rec[seq] = {'id': id, 'dataset': [], 'feats': feats}
                if len(rec) and len(rec) % 500 == 0:
                    print(f"{timer} SwissProt recording [{len(rec)}]")
            seq = ''
            feats = {}

print(f"{timer} SwissProt recording finished [{len(rec)}]")

df = pd.read_pickle(deepre)
i = 0
for line in df.values:
    seq = line[2]
    if seq in rec:
        rec[seq]['dataset'].append('DEEPre')
        i += 1
        if i % 100 == 0:
            print(f"{timer} DEEPre matching [{i}]")

print(f"{timer} DEEPre matching finished [{i}]")

i = 0
df = pd.read_pickle(ecpred)
for line in df.values:
    seq = line[1]
    if seq in rec:
        rec[seq]['dataset'].append('ECPred')
        i += 1
        if i % 200 == 0:
            print(f"{timer} ECPred matching [{i}]")

print(f"{timer} ECPred matching finished [{i}]")

for seq in list(rec.keys()):
    if not rec[seq]['dataset']:
        del rec[seq]

df = pd.DataFrame(rec)
df.to_pickle(motif_pkl)
df.to_json(motif_json)

