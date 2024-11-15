import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# from fp_sampling import cv_sample, cv_sampler_direct
from fp_sampling import cv_sample
from tokens_process.utils import read_pkl, read_map, save_pkl, counter_to_vec
import my_config as mc
import tokens_process as tkp


# test_src_path = "K:/ms_data/resources/evaluation_v44/complete_folds_smiles_holdout_1604314203.pkl"

def get_test_org():
    # 从原始pkl文件中挑选几项需要的属性
    src_path = mc.config['pkl_src_path']  # complete_folds_smiles_holdout_me.pkl
    org_path = mc.config['pkl_org_path']  # org_test_data.pkl

    if os.path.exists(org_path):
        print(f"{org_path} already exists")
        org_dict = read_pkl(org_path)
        return org_dict

    src_data = read_pkl(src_path)
    org_data = {}

    map_ls = read_map(mc.config['map_path'])
    org_data["data_information"] = src_data["data_information"]
    # spec_fp = src_data["data_fp_predicted"]
    # struct_fp = src_data["data_fp_true"]

    org_data["data_fp_predicted"] = src_data["data_fp_predicted"][:, map_ls]
    org_data["data_fp_true"] = src_data["data_fp_true"][:, map_ls]
    save_pkl(org_path, org_data)

    return org_data


def get_sampler(cv_fold):
    org_dict = get_test_org()
    data_information = org_dict["data_information"]
    spec_fps = org_dict["data_fp_predicted"]
    struct_fps = org_dict["data_fp_true"]

    db_sample_select = data_information.loc[
        ~data_information.index.str.startswith(f"fold{cv_fold}-")
    ]

    fp_true = struct_fps[db_sample_select.row_id, :]
    fp_predicted = spec_fps[db_sample_select.row_id, :]
    # print(len(fp_true), len(fp_predicted))
    sampler = cv_sample.CVSampler(fp_true, fp_predicted)
    return sampler


class PklDataset(Dataset):
    def __init__(self, option, cv_fold=None, opposite=False):
        allowed_options = {"all", 'canopus', 'gnps', 'casmi'}
        if option not in allowed_options:
            raise ValueError(f"Invalid option: {option}. Must be one of {allowed_options}")
        self.cv_fold = cv_fold
        self.cv_data = None
        self.spec_fps = None
        # self.struct_fps = None
        # 1、先选择数据集
        self.get_dataset(option)
        # 2、再挑一个 fold, cv_fold==None 全选
        self.get_cv_fold(cv_fold, opposite)
        # =======================================

    def get_dataset(self, option):
        """
        从测试数据中选出GNPS，CASMI等
        :param option: gnps, casmi, all(全选)
        :return:
        """
        # 默认全选
        sign = ""
        path = mc.config["all_path"]
        if option == "gnps":
            sign = "sirius"
            path = mc.config["gnps_path"]
        elif option == "casmi":
            sign = "casmi"
            path = mc.config["casmi_path"]
        elif option == "canopus":
            sign = "canopus"
            path = mc.config["canopus_path"]
        # 1、文件已存在
        if os.path.exists(path):
            data_dict = read_pkl(path)
            self.cv_data = data_dict["data_information"]
            self.spec_fps = data_dict["data_fp_predicted"]
            # self.struct_fps = data_dict["data_fp_true"]
            return

        # 2、文件不存在
        # 2.1、先拿到全部的数据 14047
        org_dict = get_test_org()
        data_information = org_dict["data_information"]  # 14047

        # 2.2、从data_information选出gnps或casmi对应的行 select 127 from 14047
        set_key = rf'^fold\d+-{sign}'
        db_sample_select = data_information[data_information.index.str.contains(set_key)]

        # 3、过滤掉mf中为空的值  gnps:3867--》3863
        df_filtered = db_sample_select[db_sample_select['mf'].apply(lambda x: len(x) > 0)]
        spec_fps = org_dict["data_fp_predicted"]  # 14047
        # struct_fps = org_dict["data_fp_true"]  # 14047

        fp_predicted = spec_fps[df_filtered.row_id, :]
        # fp_true = struct_fps[df_filtered.row_id, :]

        self.cv_data = df_filtered
        self.spec_fps = fp_predicted
        # self.struct_fps = fp_true

        # ========================
        data_dict = {"data_information": self.cv_data,
                     "data_fp_predicted": self.spec_fps,
                     # "data_fp_true": self.struct_fps,
                     # "data_fp_sim": self.sim_fps
                     }

        save_pkl(path, data_dict)

    def get_cv_fold(self, cv_fold, opposite):
        """
        从选出的数据集中选出 1 fold
        :param cv_fold: 0~9, None(全选)
        :return:
        """
        if cv_fold is None:
            return

        select_fold = self.cv_data.index.str.startswith(f"fold{self.cv_fold}")
        if opposite:
            select_fold = ~select_fold
        self.cv_data = self.cv_data[select_fold]

        self.spec_fps = self.spec_fps[select_fold]
        # self.struct_fps = self.struct_fps[select_fold]
        # self.sim_fps = self.sim_fps[select_fold]

    def __len__(self):
        return len(self.cv_data)
        # return len(self.grp_ok)

    def __getitem__(self, idx):
        # fp, mf, smiles_gt, smiles_lab, mf_str, smiles_str
        # fold_idx = self.cv_data.row_id
        # spec_fp = self.spec_fps[fold_idx][idx]
        spec_fp = self.spec_fps[idx]
        # sim_fp = self.sim_fps[idx]

        mf_counter = self.cv_data["mf"][idx]
        mf_vec = counter_to_vec(mf_counter)

        smiles_str = self.cv_data["smiles_canonical"][idx]
        smiles_gt, smiles_lab = tkp.smiles2_gt_lab(smiles_str)
        # smiles_gt, smiles_lab = tkp.xy_tokens_pipeline(smiles_str)

        return (torch.from_numpy(spec_fp).float(),
                torch.from_numpy(mf_vec).float(),
                torch.from_numpy(smiles_gt).float(),
                torch.from_numpy(smiles_lab).float(),
                smiles_str
                )


if __name__ == "__main__":
    # test_ls = []
    # get_sampler(cv_fold=0)
    ps = PklDataset("casmi", cv_fold=0)
    # map_ls = read_map(map_path=mc.config["map_path"])
    # ps = PklDataset("fold0-casmi")
    print(ps.__len__())
    data = ps.__getitem__(3)
    org_dict = get_test_org()
    #     print(ps.__len__())
    #     test_ls.append(ps.__len__())
    # print(test_ls)
    # right_mf = 0
    # for i in range(ps.__len__()):
    #     res = ps.__getitem__(i)[-1]
    #     if not res:
    #         print(i)
    #     right_mf += res
    # print(right_mf)
    print("HHH")
