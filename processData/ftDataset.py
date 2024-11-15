import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from fp_sampling import cv_sample
from infrastructure.score import fp_tanimoto
from tokens_process.utils import read_pkl, read_map, save_pkl, counter_to_vec
import my_config as mc
import tokens_process as tkp


# test_src_path = "K:/ms_data/resources/evaluation_v44/complete_folds_smiles_holdout_1604314203.pkl"

def get_test_org():
    # 从原始pkl文件中挑选几项需要的属性
    org_path = mc.config['pkl_org_path']
    src_path = mc.config['pkl_src_path']

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
    sampler = cv_sample.CVSampler(fp_true, fp_predicted)
    return sampler


class FTDataset(Dataset):
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
            self.struct_fps = data_dict["data_fp_true"]
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
        struct_fps = org_dict["data_fp_true"]  # 14047

        fp_predicted = spec_fps[df_filtered.row_id, :]
        fp_true = struct_fps[df_filtered.row_id, :]

        self.cv_data = df_filtered
        self.spec_fps = fp_predicted
        self.struct_fps = fp_true

        # ========================
        data_dict = {"data_information": self.cv_data,
                     "data_fp_predicted": self.spec_fps,
                     "data_fp_true": self.struct_fps,
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
        self.struct_fps = self.struct_fps[select_fold]
        # self.sim_fps = self.sim_fps[select_fold]

    def __len__(self):
        return len(self.cv_data)
        # return len(self.grp_ok)

    def __getitem__(self, idx):
        # fp, mf, smiles_gt, smiles_lab, mf_str, smiles_str
        spec_fp = self.spec_fps[idx]
        struct_fp = self.struct_fps[idx]

        mf_counter = self.cv_data["mf"][idx]
        mf_vec = counter_to_vec(mf_counter)

        smiles_str = self.cv_data["smiles_canonical"][idx]
        smiles_gt, smiles_lab = tkp.smiles2_gt_lab(smiles_str)
        # smiles_gt, smiles_lab = tkp.xy_tokens_pipeline(smiles_str)

        return (torch.from_numpy(spec_fp).float(),
                torch.from_numpy(struct_fp).float(),
                torch.from_numpy(mf_vec).float(),
                torch.from_numpy(smiles_gt).float(),
                torch.from_numpy(smiles_lab).float(),
                smiles_str
                )


if __name__ == "__main__":
    # test_ls = []
    cv_sampler = get_sampler(cv_fold=0)
    ft1 = FTDataset("all", cv_fold=0, opposite=False)
    ft1_loader = DataLoader(ft1, batch_size=256, shuffle=False, drop_last=False)
    fp_tanimoto_all = []
    for idx, data in enumerate(tqdm(ft1_loader)):
        # sim-fp 和 spec-fp
        spec_fp = data[0].round().numpy()
        struct_fp = data[1].cuda()
        sim_fp, _ = cv_sampler.sample_(struct_fp)
        sim_fp = sim_fp.round().cpu().numpy()
        tanimoto_ls = [fp_tanimoto(sim_fp[i], spec_fp[i]) for i in range(len(struct_fp))]

        # struct-fp和spec-fp
        # spec_fp = data[0].round().numpy()
        # struct_fp = data[1].cpu().numpy()
        # tanimoto_ls = [fp_tanimoto(struct_fp[i], spec_fp[i]) for i in range(len(struct_fp))]

        # struct-fp和sim-fp
        # struct_fp = data[1].cuda()
        # sim_fp, _ = cv_sampler.sample_(struct_fp)
        # sim_fp = sim_fp.round().cpu().numpy()
        # struct_fp = struct_fp.cpu().numpy()
        # tanimoto_ls = [fp_tanimoto(sim_fp[i], struct_fp[i]) for i in range(len(struct_fp))]

        fp_tanimoto_all.extend(tanimoto_ls)
    print(f"fp_tanimoto: {np.mean(fp_tanimoto_all)}")

    ft2 = FTDataset("all", cv_fold=0, opposite=False)
    ft3 = FTDataset("all", opposite=False)
    print(ft1.__len__())
    print(ft2.__len__())
    print(ft3.__len__())

    print("HHH")
