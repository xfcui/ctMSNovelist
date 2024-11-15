import torch
import numpy as np
import re
import itertools

INITIAL_CHAR = "$"
FINAL_CHAR = "*"
PAD_CHAR = "&"
SEQUENCE_LEN = 128

# The dictionary below was obtained from the cococoh dataset
# converted to selfies, then sf.get_alphabet_from_selfies


VOC = ['O',
       '=',
       'C',
       '1',
       'c',
       '2',
       '-',
       '3',
       '4',
       '(',
       ')',
       'n',
       '5',
       '6',
       '[nH]',
       'o',
       'N',
       '[N+]',
       '[O-]',
       'L',
       '[NH+]',
       'S',
       'F',
       's',
       'R',
       '#',
       'P',
       'I',
       '[N-]',
       '7',
       'p',
       '[n+]',
       '[NH3+]',
       '[C-]',
       '[NH2+]',
       '[H]']

VOC.extend([INITIAL_CHAR, FINAL_CHAR, PAD_CHAR])
# VOC_MAP = {s: i for i, s in enumerate(VOC)}
VOC2IDX = {s: i for i, s in enumerate(VOC)}
IDX2VOC = {i: s for i, s in enumerate(VOC)}

# VOC_TF = tf.convert_to_tensor(np.array(VOC))
# VOC_PT = torch.from_numpy(np.array(VOC))

# =========================对元素列表ELEMENT相关变量的定义===============================
VOC_UPPER = [x.upper() for x in VOC]

# 构建counter_matrix的元素列表, 9维, 不含H
ELEMENTS = ['C', 'F', 'I', 'L', 'N', 'O', 'P', 'R', 'S']
# Element tokens for the formula input and prediction
# 分子式的元素列表, 10维
ELEMENTS_RDKIT = ['C', 'F', 'I', 'Cl', 'N', 'O', 'P', 'Br', 'S', 'H']

elements_vec = lambda x: [e in x for e in ELEMENTS]
# (voc_size, elem_size) = (39, 9)
ELEMENT_MAP = torch.tensor(
    np.array([elements_vec(x) for x in VOC_UPPER], dtype=np.int32),
    dtype=torch.float32
)

# =========================对列表GRAMMAR相关变量的定义===============================
# (voc_size, 1) = (39, 1)
GRAMMAR = [{'(': -1, ')': 1}]
GRAMMAR_MAP = torch.tensor(
    np.stack([np.sum(np.array([[val * (key in x) for key, val in grammar_.items()] for x in VOC]), axis=1)
              for grammar_ in GRAMMAR], axis=1),
    dtype=torch.float32)
