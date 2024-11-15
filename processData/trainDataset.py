import pickle
import numpy as np
import torch
import sqlite3
from torch.utils.data import Dataset
import my_config as mc
import tokens_process as tkp
from tokens_process import counter_to_vec, fp_unpack, read_map


class FoldDataset(Dataset):
    # "fp", "mf_str", "smiles_str"----"mf_vec", "smiles_gt", "smiles_lab"
    def __init__(self, db_path, table_name, cv_fold, tv_map):
        self.db_path = db_path
        self.table_name = table_name
        self.cv_fold = cv_fold
        self.tv_map = tv_map
        self.map_ls = read_map(mc.config["map_path"])
        self.conn = None
        # self.establish_conn()

    def __len__(self):
        return len(self.tv_map)

    # def __getitem__(self, idx):
    #     self.establish_conn()
    #     # ['id', 'fold', 'fp', 'smiles_str', 'mf_str', 'mf_vec', 'smiles_gt', 'smiles_lab', 'sim_fp']
    #     # fp, mf, smiles_gt, smiles_lab, smiles_str
    #     try:
    #         id = self.tv_map[idx]
    #         sim_fp_key = mc.config['sim_fp_ls'][self.cv_fold]
    #         # sim_fp_key = "sim_fp_fold0_tf"
    #         # search_sql = f'select {sim_fp_key}, mf_vec, smiles from {self.table_name} where id=?'
    #         search_sql = f'select fp, mf_vec, smiles from {self.table_name} where id=?'
    #         self.cursor.execute(search_sql, (id,))
    #         # sim_fp, mf_bytes, smiles_str = self.cursor.fetchone()
    #         fp, mf, smiles_str = self.cursor.fetchone()
    #         self.close_conn()
    #
    #         # mf_counter = pickle.loads(mf_bytes)
    #         # mf_vec = counter_to_vec(mf_counter)
    #         mf_vec = np.frombuffer(mf, dtype=np.int32).copy()
    #         # sim_fp = np.frombuffer(sim_fp, dtype=np.float32).copy()
    #         fp = fp_unpack(fp)
    #
    #         fp = fp[self.map_ls]
    #         # sim_fp = np.unpackbits(np.frombuffer(sim_fp, dtype=np.uint8))[:3609]
    #
    #         # mf_vec = np.frombuffer(mf_vec, dtype=np.int32).copy()
    #         smiles_gt, smiles_lab = tkp.smiles2_gt_lab(smiles_str)
    #         # smiles_gt, smiles_lab = tkp.xy_tokens_pipeline(smiles_str)
    #         # smiles_lab = np.array(smiles2idx(smiles_str))[1:]
    #         # smiles_gt = smiles2vec(smiles_str)[:-1, :]
    #         # smiles_gt = np.unpackbits(np.frombuffer(smiles_gt, dtype=np.uint8).reshape(128, -1), axis=1)
    #         # smiles_lab = np.frombuffer(smiles_lab, dtype=np.int32)
    #         # sim_fp = np.frombuffer(sim_fp, dtype=np.float32)
    #
    #         return (torch.from_numpy(fp).float(),
    #                 torch.from_numpy(mf_vec).float(),
    #                 torch.from_numpy(smiles_gt).float(),
    #                 torch.from_numpy(smiles_lab).float(),
    #                 smiles_str)
    #     except Exception as e:
    #         # 处理异常，可以打印错误信息或者进行其他操作
    #         print("An error occurred:", e)
    #     finally:
    #         self.close_conn()
    def __getitem__(self, idx):
        self.establish_conn()
        try:
            id = self.tv_map[idx]
            search_sql = f'select fingerprint, mf, smiles_canonical from {self.table_name} where id=?'
            self.cursor.execute(search_sql, (id,))
            # sim_fp, mf_bytes, smiles_str = self.cursor.fetchone()
            fp, mf_bytes, smiles_str = self.cursor.fetchone()
            self.close_conn()

            mf_counter = pickle.loads(mf_bytes)
            mf_vec = counter_to_vec(mf_counter)
            fp = fp_unpack(fp)

            fp = fp[self.map_ls]
            smiles_gt, smiles_lab = tkp.smiles2_gt_lab(smiles_str)

            return (torch.from_numpy(fp).float(),
                    torch.from_numpy(mf_vec).float(),
                    torch.from_numpy(smiles_gt).float(),
                    torch.from_numpy(smiles_lab).float(),
                    smiles_str)
        except Exception as e:
            # 处理异常，可以打印错误信息或者进行其他操作
            print("An error occurred:", e)
        finally:
            self.close_conn()

    def establish_conn(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
        return self

    def close_conn(self):
        if self.conn is not None:
            self.cursor.close()
            self.conn.close()

            del self.conn
            self.conn = None
        return self
