import pickle
import re
from collections import Counter

# import molmass
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors

from tokens_process.definitions import *


def tokens_to_smiles(tokens):
    return [x.replace('L', 'Cl').replace('R', 'Br') for x in tokens]


def tokens_decode(tokens_mat, one_hot=True):
    if one_hot:
        tokens_mat = torch.argmax(tokens_mat, dim=2)

    # 将 tokens_mat 的索引映射到 VOC_TF 中的对应字符串
    tokens_ls = [[VOC[idx] for idx in sequence] for sequence in tokens_mat.tolist()]

    # 将字符串列表连接成一个字符串
    joined_tokens = [''.join(token) for token in tokens_ls]

    return joined_tokens


def tokens_crop_sequence(tokens_str, final_char=FINAL_CHAR):
    # 将 tokens_str 转换为 Python 列表并使用 final_char 分割
    tokens_split = [s.split(final_char, 1)[0] for s in tokens_str]

    return tokens_split


# # Interface for beam_model:
def ctoy(c):
    return VOC2IDX.get(c, PAD_CHAR)


y_tokens = len(VOC)


def sequence_ytoc(seq):
    seq = tokens_decode(seq, one_hot=False)
    seq = tokens_crop_sequence(seq)
    seq = tokens_to_smiles(seq)
    return seq


def embed_ytox(y):
    return F.one_hot(y.to(torch.int64), num_classes=len(VOC)).float().unsqueeze(1)


# =========================处理SMILES:生成SMILES的向量化表示===============================
def replace_smiles_ele(smiles):
    # 将smiles中的Cl和Br替换为L，R
    smiles = smiles.replace('Cl', 'L')
    smiles = smiles.replace('Br', 'R')
    return smiles


def recover_smiles_ele(smiles):
    # 将smiles中的L，R恢复为Cl和Br
    smiles = smiles.replace('L', 'Cl')
    smiles = smiles.replace('R', 'Br')
    return smiles


def spilt_smiles(smiles):
    # smiles字符串拆分成单个字符
    # 这里要使用VOC, 而不是VOC_H
    # pattern = '|'.join(map(re.escape, VOC))  # 构建匹配模式
    # matches = re.findall(pattern, smiles)  # 使用正则表达式找到字符串中的 VOC 元素
    matches_ = re.findall('[^[]|\[.*?\]', smiles)
    # if matches_ != matches:
    #     print(smiles)
    return matches_


# def pad_tokens_ls(tokens_ls):
#     # pad_ls = INITIAL_CHAR + tokens_ls + FINAL_CHAR + PAD_CHAR
#     pad_ls = [INITIAL_CHAR]
#     pad_ls.extend(tokens_ls)
#     pad_ls.append(FINAL_CHAR)
#     pad_ls = pad_ls[:SEQUENCE_LEN]  # 防止长度127的smiles越界
#     if len(pad_ls) < SEQUENCE_LEN:
#         pad_ls.extend([PAD_CHAR] * (SEQUENCE_LEN - len(pad_ls)))
#     return pad_ls

def pad_tokens_ls(tokens_ls):
    # pad_ls = INITIAL_CHAR + tokens_ls + FINAL_CHAR + PAD_CHAR
    pad_ls = [INITIAL_CHAR]
    pad_ls.extend(tokens_ls)
    pad_ls.append(FINAL_CHAR)
    # 这里不用担心 SEQUENCE_LEN - len(pad_ls) < 0
    pad_ls.extend([PAD_CHAR] * (SEQUENCE_LEN - len(pad_ls)))
    # pad_ls = pad_ls[:SEQUENCE_LEN]  # 防止长度127的smiles越界
    # if len(pad_ls) < SEQUENCE_LEN:
    #     pad_ls.extend([PAD_CHAR] * (SEQUENCE_LEN - len(pad_ls)))
    return pad_ls[:SEQUENCE_LEN]


def smiles2tokens(smiles):
    # smiles str-- > tokens (128,)
    # 向量化替换函数
    # replace_vec = np.vectorize(replace_smiles_ele)
    # smiles = str(replace_vec(smiles))
    # 把smiles中的Cl，Br替换为L，R
    smiles = replace_smiles_ele(smiles)
    tokens_ls = spilt_smiles(smiles)

    pad_ls = pad_tokens_ls(tokens_ls)
    return pad_ls


def get_atom_hydrogens(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # 如果解析失败，则返回空列表
    if mol is None:
        return []

    # 遍历分子中的每个原子，计算氢原子数目
    atom_hydrogens = []
    for atom in mol.GetAtoms():
        num_hydrogens = atom.GetTotalNumHs()
        atom_symbol = atom.GetSymbol()
        if atom_symbol == "Cl":
            atom_symbol = "L"
        if atom_symbol == "Br":
            atom_symbol = "R"
        atom_hydrogens.append((atom_symbol, num_hydrogens))
    # print(f"H nums: {sum(atom_hydrogens)}")
    return atom_hydrogens


def smiles_hydrogens(smiles):
    # smiles str-- > tokens (128,)
    # 切分smiles
    # smiles = replace_smiles_ele(smiles)
    # 计算smiles中每个原子对应的H
    atom_hydrogens = get_atom_hydrogens(smiles)

    tokens_ls = spilt_smiles(smiles.replace('Cl', 'L').replace('Br', 'R'))
    # 拼接tokens和对应的氢原子数目
    token_hydrogens = []
    atom_index = 0
    for token in tokens_ls:
        if atom_index < len(atom_hydrogens) and atom_hydrogens[atom_index][0].upper() in token.upper():
            token_hydrogens.append(atom_hydrogens[atom_index][1])
            atom_index += 1
        else:
            token_hydrogens.append(0)  # Fallback in case of mismatch

    hydrogen_pad_ls = [0]
    hydrogen_pad_ls.extend(token_hydrogens)
    hydrogen_pad_ls.append(0)
    hydrogen_pad_ls.extend([0] * (SEQUENCE_LEN - len(hydrogen_pad_ls)))
    return np.array(hydrogen_pad_ls[:SEQUENCE_LEN-1])


def tokens2idx(tokens_ls):
    return [VOC2IDX.get(c, 0) for c in tokens_ls]


def smiles2idx(smiles):
    # smiles str --> tokens (128,)-->idx (128,)
    tokens_ls = smiles2tokens(smiles)
    idx_ls = tokens2idx(tokens_ls)
    return idx_ls


def idx2vec(idx_ls):
    num_classes = len(VOC)
    return np.eye(num_classes)[np.array(idx_ls)].astype(np.float32)


def tokens2vec(tokens_ls):
    idx_ls = tokens2idx(tokens_ls)
    return idx2vec(idx_ls)


def smiles2vec(smiles):
    # smiles --> tokens -->idx --> one hot vec
    idx_ls = smiles2idx(smiles)
    return idx2vec(idx_ls)


# def smiles2_gt_lab(smiles):
#     # gt: $ smiles [*]
#     # lab: smiles * & &...
#     gt_ls = smiles2idx(smiles)  # (128,)
#     lab_ls = gt_ls[1:]
#     lab_ls.append(VOC2IDX.get(PAD_CHAR))  # (128,)
#
#     num_classes = len(VOC)
#     gt = np.eye(num_classes)[np.array(gt_ls)].astype(np.int32)
#     return gt, np.array(lab_ls, dtype=np.int32)  # (seq_len, voc_size), (seq_len,)
def smiles2_gt_lab(smiles):
    # gt: $ smiles [*]
    # lab: smiles * & &...
    idx_ls = smiles2idx(smiles)  # (128,)
    gt_ls = idx_ls[:-1]
    lab_ls = idx_ls[1:]
    gt = idx2vec(gt_ls)  # (127, 39)
    return gt, np.array(lab_ls, dtype=np.float32)  # (seq_len, voc_size), (seq_len,)


def xy_tokens_pipeline(smiles):
    idx_ls = smiles2idx(smiles)
    sm_vec = idx2vec(idx_ls)  # (128, 39)
    x_vec = sm_vec[:-1, :]  # (127, 39)
    y_vec = sm_vec[1:, :]  # (127, 39)
    return x_vec, y_vec


# =========================处理SMILES:从向量化表示中解析出SMILES===============================
def remove_pad(sm):
    # 删除填充单个smiles的特殊符号
    if sm[0] == INITIAL_CHAR:
        sm = sm[1:]
    end_index = sm.find(FINAL_CHAR)
    if end_index != -1:
        sm = sm[:end_index]
    return sm


def vec2idx(sm_vec):
    # sm_vec (128, 39) --> sm_idx (128,)
    sm_idx = np.argmax(sm_vec, axis=1)
    return sm_idx


def tokens2smiles(tokens_ls):
    sm_pad = ''.join(t for t in tokens_ls)
    smiles = remove_pad(sm_pad)
    # 向量化替换函数
    replace_vec = np.vectorize(recover_smiles_ele)
    smiles = str(replace_vec(smiles))  # 把smiles中的L，R替换为Cl，Br
    return smiles


def idx2smiles(sm_idx):
    # sm_idx (128,) --> smiles
    sm_pad = ''.join(VOC[idx] for idx in sm_idx.tolist())
    smiles = remove_pad(sm_pad)

    # 向量化替换函数
    replace_vec = np.vectorize(recover_smiles_ele)
    smiles = str(replace_vec(smiles))  # 把smiles中的L，R替换为Cl，Br
    return smiles


def vec2smiles(sm_vec):
    # sm_vec (128, 39) --> smiles
    sm_idx = vec2idx(sm_vec)
    smiles = idx2smiles(sm_idx)
    return smiles


def vec2smiles_batch(sm_vec, sm_lab):
    seq = tokens_decode(sm_vec, one_hot=True)
    seq = tokens_crop_sequence(seq)
    smiles_ls = tokens_to_smiles(seq)
    # sms_idx = np.argmax(sm_vec, axis=-1)
    # smiles_ls = [idx2smiles(idx_ls) for idx_ls in sms_idx]
    # print("HHH")
    # sm_num = np.sum(np.array(smiles_ls) == np.array(sm_lab))
    # valid_num = sum([Chem.MolFromSmiles(i)!=None for i in smiles_ls])
    return smiles_ls


def get_formula(m, h=True):
    m_ = Chem.AddHs(m)
    c = Counter(atom.GetSymbol() for atom in m_.GetAtoms())
    if not h:
        c['H'] = 0
    return c


if __name__ == "__main__":
    # smiles = 'O=C([O-])C(CCCCCC1C=CC(CCCCC)CC1O)C(O)CCC2(O)CC(Cc3cc[nH+]c(N)c3)C4(CCCC4)C2O'
    smiles = "O=C(OC)C(NC(=O)C(NC(=O)N1CCCC1COc2nc(cs2)-c3cnccc3)C(C)C)CC4N3"
    # H_ls = smiles_hydrogens(smiles)
    # # x1, y1 = xy_tokens_pipeline(smiles)
    # # x2, y2 = smiles2_gt_lab(smiles)
    # print(H_ls)
    # print('HHH')
