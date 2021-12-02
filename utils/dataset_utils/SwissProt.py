from utils.env import *
from lxml import etree

data_root = '../../dataset/SwissProt'
data_file = os.path.join(data_root, 'uniprot_sprot.xml')
dataset_file = os.path.join(data_root, 'swissprot_ixs.npy')


def prepare_dataset(safe_mode=True):
    if safe_mode and os.path.exists(dataset_file):
        print('Dataset file already exists, use safe_mode=False to overwrite.')
        return False
    context = etree.iterparse(str(data_file), events=["end"], tag="{http://uniprot.org/uniprot}entry", huge_tree=True)
    data = []

    for _, elem in tqdm(iter(context)):
        seqs = elem.findall("{http://uniprot.org/uniprot}sequence")
        for seq in seqs:
            s = seq.text
            if s and str(s) != "None":
                data.append([token2ix[e] for e in s])
        elem.clear()

    random.shuffle(data)
    np.save(dataset_file, data)
    print(f"Dataset file prepared with total {len(data)} sequences.")
    return True


if __name__ == '__main__':
    prepare_dataset()
