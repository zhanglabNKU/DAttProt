import random
import os
import numpy as np
from tqdm import tqdm
random.seed(1024)
vocab_size = 23  # excluding PADDING, B, Z
# X: MASK, B: D or N, Z: E or Q
tokens = [
    'PADDING',
    'L', 'A', 'G', 'V', 'S',
    'E', 'I', 'K', 'R', 'D',
    'T', 'P', 'N', 'Q', 'F',
    'Y', 'M', 'H', 'C', 'W',
    'U', 'O', 'X', 'B', 'Z'
]
token2ix = {name: i for i, name in enumerate(tokens)}
dataset_root = 'dataset/'
model_root = 'saved_models/'
if not os.path.exists(model_root):
    os.makedirs(model_root)
