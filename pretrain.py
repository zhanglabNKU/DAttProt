cuda_id = 0  # GPU id
model_num = 0  # number the model
train_rate, mask_rate = 0.9, 0.15  # the ratio of training samples to all samples, mask ratio in Masked LM
train_ep, save_ep, batch_sz = 300, 20, 32  # max training epoches, epoch interval to save checkpoints, batch size
warmup_it, log_it = 1e4, 1e3  # max warm-up iterations, iteration interval to write into the log file
lr = 5e-5  # learning rate
optim_mode = 'noam'  # optimizer: noam or adam
mask_flag = True  # if masking padding tokens

args = {
    'seq_len': 512,  # sequence length
    'embedding_dim': 16,  # dimension of embedding
    'feature_dim': 512,  # dimension of Transformer hidden features
    'encoding_dim': 16,  # dimension of the multi-scale convolutions output
    'nhead': 8,  # number of heads in Transformer encoders
    'nlayer': 6,  # number of Transformer encoder layers
    'dropout': 0.1,  # dropout probability
    'key_padding_mask': mask_flag
}

from model.xfmr_att_fc import setup_seed, DAttProt_lm, RandMask
from utils.loader import UnsupervisedDataset
from utils.optim import NoamOpt, rate
from utils.timer import Timer
from utils.plot import plot_pretrain
from utils.env import *
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

setup_seed(1024)

device = torch.device(f"cuda:{cuda_id}")
print('torch version', torch.__version__)

dataset_file = 'dataset/SwissProt/swissprot_ixs.npy'
model_root = 'saved_models/'
log_name = f"log_pretrain_{model_num}.txt"
if os.path.exists(log_name):
    os.remove(log_name)

print('Loading pre-training dataset...')
dataset = UnsupervisedDataset(dataset_file, args['seq_len'])
train_sz = int(len(dataset) * train_rate)
trainset = Subset(dataset, range(train_sz))
validset = Subset(dataset, range(train_sz, len(dataset)))
trainloader = DataLoader(trainset, batch_sz)
validloader = DataLoader(validset, batch_sz)
print('Pre-training dataset loaded.')

args['vocab_size'] = vocab_size
timer = Timer()
masker = RandMask(vocab_size, pred_rate=mask_rate)
net = DAttProt_lm(**args).to(device)
net.train()
if optim_mode == 'adam':
    optimizer = optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.98))
else:
    adam = optim.AdamW(net.parameters(), lr=0, betas=(0.9, 0.98))
    optimizer = NoamOpt(args['feature_dim'], lr, warmup_it, adam)
criterion = nn.CrossEntropyLoss()


if __name__ == '__main__':
    sample_sz = ep = it = 0
    timer.start()
    it_list, lr_list, trn_list, val_list, acc_list = [], [], [], [], []
    while ep < train_ep:
        for seq_ixs, lengths in trainloader:
            inputs, pred_pos, targets = masker(seq_ixs, lengths)
            inputs, pred_pos, targets = inputs.to(device), pred_pos.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = net.pretrain(inputs, pred_pos)
            trn_loss = criterion(outputs, targets)
            trn_loss.backward()
            optimizer.step()
            sample_sz += len(seq_ixs)
            it += 1
            if it and it % log_it == 0:
                net.eval()
                total, correct = 0, 0
                avg_loss = 0.

                with torch.no_grad():
                    for seq_ixs, lengths in validloader:
                        inputs, pred_pos, targets = masker(seq_ixs, lengths)
                        inputs, pred_pos, targets = inputs.to(device), pred_pos.to(device), targets.to(device)
                        outputs = net.pretrain(inputs, pred_pos)
                        val_loss = criterion(outputs, targets)
                        avg_loss += val_loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()
                    avg_loss /= len(validloader)
                log_info = f"ep:{ep}/{train_ep} it:{it / 1e3:.1f}K sz:{sample_sz / 1e3:.1f}K "
                log_info += f"| bz:{batch_sz} lr:{rate(optimizer):.2e} tm:{timer}\n"
                log_info += f"    trn:{trn_loss:.4f} val:{avg_loss:.4f} acc:{100 * correct / total:.2f}%"
                print(log_info)
                it_list.append(it / 1e3)
                lr_list.append(rate(optimizer) * 1e5)
                trn_list.append(trn_loss)
                val_list.append(val_loss)
                acc_list.append(100 * correct / total)
                with open(log_name, "a") as f:
                    print(log_info, file=f)
                net.train()
        ep += 1

        pretrain_model = {
            'optim_mode': optim_mode,
            'args': args,
            'sz_pre': sample_sz,
            'model_pre': net.base_dict(),
            'optimizer_pre': optimizer.state_dict()
        }
        if ep % save_ep == 0 or ep >= train_ep:
            model_name = f"pretrain_ly{args['nlayer']}_ep{ep}of{train_ep}_{model_num}"
            torch.save(pretrain_model, os.path.join(model_root, f"{model_name}.pkl"))
            plot_pretrain(it_list, lr_list, acc_list, trn_list, val_list,
                          save=os.path.join(model_root, f"{model_name}.png"))
            print(f"Pretrain models of ep{ep} saved.")
