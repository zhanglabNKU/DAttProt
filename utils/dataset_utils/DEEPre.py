from utils.env import *
import pandas as pd
from collections import defaultdict

data_root = '../../dataset/DEEPre'
data_file = os.path.join(data_root, 'DEEPre_data.pkl')
dataset_files = {
    'ixs': os.path.join(data_root, 'deepre_ixs.npy'),
    'label2ix': os.path.join(data_root, 'deepre_label2ix.npy'),
    'labelcount': os.path.join(data_root, 'deepre_labelcount.npy'),
}


def prepare_dataset(safe_mode=True):
    if safe_mode and os.path.exists(dataset_files['ixs']):
        print('Dataset files already exist, set safe_mode=False to overwrite files.')
        return False
    df = pd.read_pickle(data_file)
    data = []
    sublabel = defaultdict(dict)
    label2ix = {
        (0,): 0,
        (0, 0): 0,
        (0, 0, 0): 0,
        (1,): 1,
        (1, 1): 0,
        (1, 2): 1,
        (1, 3): 2,
        (1, 4): 3,
        (1, 5): 4,
        (1, 6): 5
    }
    labelcount = defaultdict(int)
    labelcount[(1,)] = 6
    for _, class_info, seq, _ in tqdm(df.values):
        if not isinstance(class_info, str):
            label = [0, 0, 0]
        else:
            classes = class_info.split('.')
            c1, c2 = int(classes[0]), int(classes[1])
            if c2 not in sublabel[c1]:
                sublabel[c1][c2] = len(sublabel[c1])
                labelcount[(1, c1)] += 1
            l = sublabel[c1][c2]
            label2ix[(1, c1, c2)] = l
            label = [1, c1 - 1, l]
        data.append([token2ix[e] for e in seq] + label)
    random.shuffle(data)
    np.save(dataset_files['ixs'], data)
    np.save(dataset_files['label2ix'], label2ix)
    np.save(dataset_files['labelcount'], labelcount)
    print(f"Dataset files prepared with total {len(data)} sequences")
    return True


if __name__ == '__main__':
    prepare_dataset()
