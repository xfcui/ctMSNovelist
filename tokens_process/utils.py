import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tokens_process.definitions import *


# =========================处理分子指纹FP===============================
def read_map(map_path):
    data = pd.read_csv(map_path, delimiter='\t')
    data.set_index("absoluteIndex", inplace=True)
    map_ls = data.index.tolist()
    return map_ls


def fp_unpack(fp):
    # fp: bytes --> np: (8925,)
    fp_decoded = torch.tensor(list(bytearray(fp)), dtype=torch.uint8)  # (4,)
    fp_decoded = fp_decoded.unsqueeze(1)  # (4,1)

    # 这里每个元素表示一个字节中的一个位（从低位到高位）
    bits = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8).view(1, -1)  # (1, 8)

    fp_unpacked = torch.reshape(torch.bitwise_and(fp_decoded, bits), (-1,))

    # 将非零值（即二进制中的 1）转换为 uint8 类型的 1，零值保持为 0
    fp_unpacked = (fp_unpacked != 0).to(torch.uint8)

    # 返回解码后的指纹张量，截取前 8925 位
    return fp_unpacked[:8925].numpy()


def fp_pipeline_unpack(fp):
    # 将多个bytes 的fp 转为np: (batch_size, 8925)
    # fp 的形状为 [batch_size]，解码后 fp_decoded 的形状为 [batch_size, num_bytes]
    fp_decoded = torch.stack([torch.tensor(list(bytearray(f)), dtype=torch.uint8) for f in fp])  # (1,4)

    # 创建一个包含二进制位掩码的常量张量
    # 这里每个元素表示一个字节中的一个位（从低位到高位）
    bits = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8).view(1, 1, -1)  # (1,1,8)

    # 在 fp_decoded 的最后一维增加一个维度以进行位与操作
    # 扩展维度后 fp_decoded 的形状为 [batch_size, num_bytes, 1]
    fp_decoded = fp_decoded.unsqueeze(2)  # (1,4,1)

    # 计算位与操作，得到每个字节中的各个位
    # 使用位掩码将每个字节转换为多个二进制位
    # 计算后的形状为 [batch_size, num_bytes, 8]，然后再 reshape 为 [batch_size, num_bits]
    fp_unpacked = torch.reshape(torch.bitwise_and(fp_decoded, bits), (fp_decoded.shape[0], -1))

    # 将非零值（即二进制中的 1）转换为 uint8 类型的 1，零值保持为 0
    fp_unpacked = (fp_unpacked != 0).to(torch.uint8)

    # 返回解码后的指纹张量，截取前 8925 位
    return fp_unpacked[:, :8925].numpy()


# =========================处理分子式MF===============================
def is_valid_molecule(smiles):
    # 1. 检查是否可以解析为分子对象
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # print(f"invalid smiles 1: {smiles}")
        return False

    # 2. 检查 SMILES 长度是否超过 127 个字符
    if len(smiles) > 127:
        # print(f"invalid smiles 2: {smiles}")
        return False

    # 3. 检查是否包含断开连接的 SMILES
    if '.' in smiles:
        # print(f"invalid smiles 3: {smiles}")
        return False

    # 4. 检查分子量是否大于 1000 Da
    mol_weight = Chem.Descriptors.MolWt(mol)
    if mol_weight > 1000:
        # print(f"invalid smiles 4: {smiles}")
        return False

    # 5. 检查是否有电荷
    if Chem.GetFormalCharge(mol) != 0:
        # print(f"invalid smiles 5: {smiles}")
        return False

    # 6. 检查是否超过 7 个环
    ring_info = mol.GetRingInfo()
    num_rings = ring_info.NumRings()
    if num_rings > 7:
        # print(f"invalid smiles 6: {smiles}")
        return False

    # 7. 检查是否包含 C、H、N、O、P、S、Br、Cl、I 和 F 以外的元素
    elements = set([atom.GetSymbol() for atom in mol.GetAtoms()])
    valid_elements = {'C', 'H', 'N', 'O', 'P', 'S', 'Br', 'Cl', 'I', 'F'}
    if not elements.issubset(valid_elements):
        # print(f"invalid smiles 7: {smiles}")
        return False

    return True


def counter_to_str(counter):
    # 把counter类型的分子式转化为str
    formula_str = ''
    for element, count in sorted(counter.items()):
        if count == 1:
            formula_str += element
        else:
            formula_str += f"{element}{count}"
    return formula_str


def counter_to_vec(counter):
    return np.array([counter[el] for el in ELEMENTS_RDKIT], dtype=np.int32)


# =========================读写文件===============================
def read_pkl(file_path):
    with open(file_path, 'rb') as file:
        # res = pickle.load(file)
        print(file_path)
        res = pd.read_pickle(file)
    return res


def save_pkl(file_path, data):
    with open(file_path, "wb") as file:
        # pickle.dump(data, file)
        pd.to_pickle(data, file)


if __name__ == "__main__":
    # smiles = "NC(=O)C=Cc1ccccc1"
    smiles = 'O=C([O-])C(CCCCCC1C=CC(CCCCC)CC1O)C(O)CCC2(O)CC(Cc3cc[nH+]c(N)c3)C4(CCCC4)C2O'
    # spilt_smiles有问题待解决
    # l = spilt_smiles(smiles)
    # file_path = "/home/sf123/MS_DATA/complete_folds_smiles_holdout_me.pkl"
    # read_pkl(file_path)
    # ls = smiles2vec(smiles)
    # sm = vec2smiles(ls)
    # print(sm)
    # print(sm == smiles)
    # mf = "C1F2I3Cl4N5O6P7Br8S9H10"
    # vec = mf2vec(mf)
    # print(vec)

# =========================处理SMILES:生成SMILES的向量化表示===============================
#
# def replace_smiles_ele(smiles):
#     # 将smiles中的Cl和Br替换为L，R
#     smiles = smiles.replace('Cl', 'L')
#     smiles = smiles.replace('Br', 'R')
#     return smiles
#
#
# def recover_smiles_ele(smiles):
#     # 将smiles中的L，R恢复为Cl和Br
#     smiles = smiles.replace('L', 'Cl')
#     smiles = smiles.replace('R', 'Br')
#     return smiles
#
#
# def spilt_smiles(smiles):
#     # smiles字符串拆分成单个字符
#     # 这里要使用VOC, 而不是VOC_H
#     # pattern = '|'.join(map(re.escape, VOC))  # 构建匹配模式
#     # matches = re.findall(pattern, smiles)  # 使用正则表达式找到字符串中的 VOC 元素
#     matches_ = re.findall('[^[]|\[.*?\]', smiles)
#     # if matches_ != matches:
#     #     print(smiles)
#     return matches_
#
#
# def pad_tokens_ls(tokens_ls):
#     # pad_ls = INITIAL_CHAR + tokens_ls + FINAL_CHAR + PAD_CHAR
#     pad_ls = [INITIAL_CHAR]
#     pad_ls.extend(tokens_ls)
#     pad_ls.append(FINAL_CHAR)
#     pad_ls = pad_ls[:SEQUENCE_LEN]  # 防止长度127的smiles越界
#     if len(pad_ls) < SEQUENCE_LEN:
#         pad_ls.extend([PAD_CHAR] * (SEQUENCE_LEN - len(pad_ls)))
#     return pad_ls
#
#
# def smiles2tokens(smiles):
#     # smiles str-- > tokens (128,)
#     # 向量化替换函数
#     replace_vec = np.vectorize(replace_smiles_ele)
#     smiles = str(replace_vec(smiles))  # 把smiles中的Cl，Br替换为L，R
#     tokens_ls = spilt_smiles(smiles)
#
#     pad_ls = pad_tokens_ls(tokens_ls)
#     return pad_ls
#
#
# def tokens2idx(tokens_ls):
#     return [VOC2IDX.get(c, 0) for c in tokens_ls]
#
#
# def smiles2idx(smiles):
#     # smiles str --> tokens (128,)-->idx (128,)
#     tokens_ls = smiles2tokens(smiles)
#     idx_ls = tokens2idx(tokens_ls)
#     return idx_ls
#
#
# def idx2vec(idx_ls):
#     num_classes = len(VOC)
#     return np.eye(num_classes)[np.array(idx_ls)].astype(np.int32)
#
#
# def tokens2vec(tokens_ls):
#     idx_ls = tokens2idx(tokens_ls)
#     return idx2vec(idx_ls)
#
#
# def smiles2vec(smiles):
#     # smiles --> tokens -->idx --> one hot vec
#     idx_ls = smiles2idx(smiles)
#     return idx2vec(idx_ls)
#
#
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
#
#
# # =========================处理SMILES:从向量化表示中解析出SMILES===============================
# def remove_pad(sm):
#     # 删除填充单个smiles的特殊符号
#     if sm[0] == INITIAL_CHAR:
#         sm = sm[1:]
#     end_index = sm.find(FINAL_CHAR)
#     if end_index != -1:
#         sm = sm[:end_index]
#     return sm
#
#
# def vec2idx(sm_vec):
#     # sm_vec (128, 39) --> sm_idx (128,)
#     sm_idx = np.argmax(sm_vec, axis=1)
#     return sm_idx
#
#
# def tokens2smiles(tokens_ls):
#     sm_pad = ''.join(t for t in tokens_ls)
#     smiles = remove_pad(sm_pad)
#     # 向量化替换函数
#     replace_vec = np.vectorize(recover_smiles_ele)
#     smiles = str(replace_vec(smiles))  # 把smiles中的L，R替换为Cl，Br
#     return smiles
#
#
# def idx2smiles(sm_idx):
#     # sm_idx (128,) --> smiles
#     sm_pad = ''.join(VOC[idx] for idx in sm_idx.tolist())
#     smiles = remove_pad(sm_pad)
#
#     # 向量化替换函数
#     replace_vec = np.vectorize(recover_smiles_ele)
#     smiles = str(replace_vec(smiles))  # 把smiles中的L，R替换为Cl，Br
#     return smiles
#
#
# def vec2smiles(sm_vec):
#     # sm_vec (128, 39) --> smiles
#     sm_idx = vec2idx(sm_vec)
#     smiles = idx2smiles(sm_idx)
#     return smiles
#
#
# def vec2smiles_batch(sm_vec, sm_lab):
#     sms_idx = np.argmax(sm_vec, axis=-1)
#     smiles_ls = [idx2smiles(idx_ls) for idx_ls in sms_idx]
#     # print("HHH")
#     # valid_num = np.array([Chem.MolFromSmiles(s) != None for s in smiles_ls]).sum()
#     sm_num = np.sum(np.array(smiles_ls) == np.array(sm_lab))
#     valid_num = sum([Chem.MolFromSmiles(i) != None for i in smiles_ls])
#     return smiles_ls
