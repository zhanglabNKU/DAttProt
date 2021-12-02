cuda_id = 0  # GPU id
model_num = 0  # number the model

pretrain_name = f"pretrain_ly6_ep300of300_{model_num}.pkl"  # pre-trained model file name
scales = (5, 10, 20)  # sizes of multi-scale convolutional kernels
max_len = 512  # sequence length
encoder_nlayer = 6  # number of encoder layers in use

freeze_nlayer = 0  # number of encoder layers frozen, default: 0

lr, wd = 1e-5, 1e-4  # learning rate, weight decay
batch_sz, train_ep = 128, 100  # batch size, max training epoches
decay_it, lr_decay = 1e3, 0.95  # learning rate decay iteration, learning rate decay rate

level = 0  # task level: 0 or 1

import copy
import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from model.xfmr_att_fc import setup_seed, DAttProt_cls
from utils.env import *
from utils.loader import SupervisedDataset
from utils.optim import rate, freeze, set_rate
from utils.plot import plot_ecpred
from utils.timer import Timer
from torch.utils.data import DataLoader

setup_seed(1024)

device = torch.device(f"cuda:{cuda_id}")
print('torch version', torch.__version__)

class_head = f"lv{level}"
model_name = f"ECPred_{class_head}_ly{encoder_nlayer}_{model_num}"
dataset_files = {
    'ixs': dataset_root + 'ECPred/ecpred_ixs.npy',
    'label2ix': dataset_root + 'ECPred/ecpred_label2ix.npy',
    'labelcount': dataset_root + 'ECPred/ecpred_labelcount.npy'
}
log_name = "log_ecpred.txt"
if os.path.exists(log_name):
    os.remove(log_name)

print('Loading training dataset...')
labelcount: dict = np.load(dataset_files['labelcount'], allow_pickle=True).item()
if level:
    nclass = labelcount[1]
else:
    nclass = 2
train_data = []
test_data = []
for line in np.load(dataset_files['ixs'], allow_pickle=True):
    # line: [seq, main_class_label, sub_class_label, train_test]
    if level and line[-3] == 0:
        continue
    label = line[level - 3]
    if line[-1]:
        test_data.append(np.append(line[:-3], label))
    else:
        train_data.append(np.append(line[:-3], label))

trainloader = DataLoader(SupervisedDataset(train_data, max_len), batch_sz)
testloader = DataLoader(SupervisedDataset(test_data, max_len), batch_sz)

pre_model = torch.load(os.path.join(model_root, pretrain_name), map_location=f"cuda:{cuda_id}")
args = pre_model['args']
assert freeze_nlayer <= encoder_nlayer <= args['nlayer']
args['scales'] = scales
args['train_layer'] = encoder_nlayer
args['seq_len'] = max_len
args['nclass'] = nclass
args['key_padding_mask'] = True
timer = Timer()
net = DAttProt_cls(**args).to(device)
net.load_base_dict(pre_model['model_pre'])
# vocab = torch.LongTensor(range(vocab_size)).to(device)
# print(net.embedding(vocab))
if freeze_nlayer:
    freeze([net.embedding, net.pos_embedding, net.map,
            net.coder.seq_embed[:freeze_nlayer]])
    optimizer = AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=lr, betas=(0.9, 0.98), weight_decay=wd)
else:
    optimizer = AdamW(net.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=wd)
if lr_decay > 0:
    scheduler = lr_scheduler.StepLR(optimizer, int(decay_it), lr_decay)
else:
    scheduler = None
criterion = nn.CrossEntropyLoss()


def train():
    global samples_sz, it
    net.train()
    train_loss = 0.
    total = correct = 0

    for seq_ixs, labels in trainloader:
        inputs, targets = seq_ixs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        train_loss += loss.item() / len(trainloader)
        with torch.no_grad():
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            samples_sz += len(labels)
    it += len(trainloader)

    return train_loss, 100. * correct / total


def test():
    net.eval()
    test_loss = 0.
    total = correct = 0
    pos = [0] * nclass
    TP = [0] * nclass
    P = [0] * nclass

    with torch.no_grad():
        for seq_ixs, labels in testloader:
            inputs, targets = seq_ixs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += (loss.item()) / len(testloader)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            for i in range(nclass):
                TP[i] += ((predicted == targets) * (targets == i)).sum().item()
                pos[i] += (targets == i).sum().item()
                P[i] += (predicted == i).sum().item()
    pre = [TP[i] / max(P[i], 1) for i in range(nclass)]
    rec = [TP[i] / max(pos[i], 1) for i in range(nclass)]
    f1 = [(2 * pre[i] * rec[i]) / max(pre[i] + rec[i], 1e-4) for i in range(nclass)]
    pre = sum(pre) / nclass if nclass > 2 else pre[-1]
    rec = sum(rec) / nclass if nclass > 2 else rec[-1]
    f1 = sum(f1) / nclass if nclass > 2 else f1[-1]
    return test_loss, 100. * correct / total, pre, rec, f1


if __name__ == '__main__':
    loss_list, acc_list, pr_list, f1_list, epoch_list = [], [], [], [], []
    # samples_sz = pre_model['sz_pre']
    # train_sz += samples_sz
    samples_sz = ep = it = 0
    best_f1 = 0.
    best_model = {}

    print('-' * 20, ' Fine-Turning ', '-' * 20)
    timer.start()

    while ep < train_ep:
        epoch_list.append(ep)
        train_loss, train_acc = train()
        test_loss, test_acc, pre, rec, f1 = test()
        loss_list.append([train_loss, test_loss])
        acc_list.append([train_acc, test_acc])
        pr_list.append([pre, rec])
        f1_list.append(f1)
        log_info = f"ep:{ep}/{train_ep} it:{it / 1e3:.1f}K sz:{samples_sz / 1e6:.2f}M "
        log_info += f"| bz:{batch_sz} lr:{rate(optimizer):.2e} tm:{timer}\n"
        log_info += f"    trn loss:{train_loss:.4f} acc:{train_acc:.2f}%\n"
        log_info += f"    tst loss:{test_loss:.4f} acc:{test_acc:.2f}% p:{pre:.3f} r:{rec:.3f} f1:{f1:.3f}"
        print(log_info)
        with open(log_name, "a") as f:
            print(log_info, file=f)

        if f1 > best_f1:
            best_f1 = f1
            best_model = {'p': pre,
                          'r': rec,
                          'f1': f1,
                          'args': args,
                          'model': copy.deepcopy(net.state_dict())}
            print('*' * 10)
        ep += 1

    torch.save(best_model, os.path.join(model_root, f"{model_name}.pkl"))
    plot_ecpred(epoch_list, loss_list, acc_list, pr_list, f1_list,
                save=os.path.join(model_root, f"{model_name}.png"))
    print(f"Model of ec level {level} saved.")
