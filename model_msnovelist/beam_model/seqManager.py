import numpy as np


class SequenceManager:
    def __init__(self, n, k, steps, final_char, pad_char):
        self.n = n
        self.k = k
        self.steps = steps
        self.final_char = final_char
        self.complete_seqs_mat = np.full((self.n * self.k, self.steps + 1), pad_char, dtype=object)
        self.complete_scores_mat = np.full((self.n * self.k,), -np.inf, dtype=float)

    def add_sequence(self, seq, score, ind):
        i = ind // self.k
        start = i * self.k
        end = start + self.k
        # 在指定区间内寻找插入位置
        idx = start + np.searchsorted(-self.complete_scores_mat[start:end], -score)
        if idx < end:  # 确保不会越界
            # 向后移动元素以空出插入位置
            self.complete_scores_mat[idx+1:end] = self.complete_scores_mat[idx:end-1]
            self.complete_seqs_mat[idx+1:end] = self.complete_seqs_mat[idx:end-1]
            # 插入新元素
            self.complete_scores_mat[idx] = score
            self.complete_seqs_mat[idx][:seq.shape[0]] = seq
            # print("HHH")

    def print_mats(self):
        print("complete_seqs_mat:")
        print(self.complete_seqs_mat.reshape(self.n, self.k))
        print("complete_scores_mat:")
        print(self.complete_scores_mat.reshape(self.n, self.k))

    def sequences2smiles(self):
        complete_seqs_mat = self.complete_seqs_mat[:, 1:]
        complete_seqs_mat = np.array([[word for word in seq] for seq in complete_seqs_mat])
        seqs = [''.join(seq) for seq in complete_seqs_mat]
        seqs = [seq.split(self.final_char, 1)[0] for seq in seqs]
        smiles = [x.replace('L', 'Cl').replace('R', 'Br') for x in seqs]
        return smiles, self.complete_scores_mat.reshape(self.n, self.k)
# n, k = 2, 2
# pad_char = '<PAD>'
# manager = SequenceManager(n, k, pad_char)
#
# # 假设每个时刻会生成以下seq和score
# seqs = ['seq1', 'seq2', 'seq3', 'seq4', 'seq5']
# scores = [0.9, 0.5, 0.8, 0.7, 1.0]
# inds = [0, 0, 1, 1, 0]
#
# for seq, score, ind in zip(seqs, scores, inds):
#     manager.add_sequence(seq, score, ind)
#
# manager.print_mats()
