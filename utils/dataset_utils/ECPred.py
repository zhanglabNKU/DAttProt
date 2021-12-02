from utils.env import *
import pandas as pd

data_root = '../../dataset/ECPred'
data_file = os.path.join(data_root, 'ECPred_data.pkl')
dataset_files = {
    'ixs': os.path.join(data_root, 'ecpred_ixs.npy'),
    'label2ix': os.path.join(data_root, 'ecpred_label2ix.npy'),
    'labelcount': os.path.join(data_root, 'ecpred_labelcount.npy'),
}


def prepare_dataset(safe_mode=True):
    if safe_mode and os.path.exists(dataset_files['ixs']):
        print('Dataset files already exist, set safe_mode=False to overwrite files.')
        return False
    df = pd.read_pickle(data_file)
    data = []
    labelcount = {1: 6}
    label2ix = {
        (0,): 0,
        (0, 0): 0,
        (1,): 1,
        (1, 1): 0,
        (1, 2): 1,
        (1, 3): 2,
        (1, 4): 3,
        (1, 5): 4,
        (1, 6): 5
    }
    for df_line in tqdm(df.values):
        seq, ec_info, traintest = df_line[1:4]
        if len(ec_info) == 0:
            label = [0, 0, traintest]
        else:
            label = [1, int(ec_info[0].split('.')[0]) - 1, traintest]
        data.append([token2ix[e] for e in seq] + label)
    random.shuffle(data)
    np.save(dataset_files['ixs'], data)
    np.save(dataset_files['label2ix'], label2ix)
    np.save(dataset_files['labelcount'], labelcount)
    print(f"Dataset files prepared with total {len(data)} sequences")
    return True


if __name__ == '__main__':
    prepare_dataset()
