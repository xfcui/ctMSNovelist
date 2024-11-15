import pickle
import sqlite3
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import my_config as mc
from processData.trainDataset import FoldDataset
from tokens_process.utils import is_valid_molecule, read_map, fp_unpack, \
    fp_pipeline_unpack, counter_to_vec


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # =右边"0,1",代表使用标号为0,和1的GPU


class SQLiteHelper:
    def __init__(self, db_path):
        self.SCHEMA_DEF = {
            'compounds':
                (
                    ('id', 'INTEGER PRIMARY KEY'),
                    ('fp', 'BLOB'),  # np.float32
                    ('smiles', 'CHAR(128)'),  # str
                    ('mf', 'BLOB'),  # bytes counter
                    ('mf_vec', 'BLOB'),  # np.int32
                    ('grp', 'CHAR(128)'),
                    ('perm_order', 'INT')
                )
        }

        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def create_schema(self):
        for (table_name, table_schema) in self.SCHEMA_DEF.items():
            self._create_table(table_name, table_schema)

    def _create_table(self, table_name, table_schema):
        sql_str = 'CREATE TABLE IF NOT EXISTS '
        sql_str = sql_str + table_name + ' ('
        table_spec = [' '.join(table_tpl) for table_tpl in table_schema]
        sql_str = sql_str + ', '.join(table_spec) + ' )'
        self.execute(sql_str)

    def execute(self, sql, value=None):
        # 无返回值
        if value:
            self.cursor.execute(sql, value)
        else:
            self.cursor.execute(sql)

    def execute_rtn(self, sql, value=None):
        # 有返回值
        if value:
            self.cursor.execute(sql, value)
        else:
            self.cursor.execute(sql)
        data = self.cursor.fetchall()
        return data

    def read(self, sql, size=1000, key_ls=mc.config["key_ls"]):
        self.cursor.execute(sql, key_ls)
        if size == 123:
            data = self.cursor.fetchall()  # [:size]
        else:
            data = self.cursor.fetchmany(size)
        return data

    def read_batch(self, sql, batch_size, offset):
        self.cursor.execute(sql, (batch_size, offset))
        batch_data = self.cursor.fetchall()
        return batch_data

    def read_batch_fold(self, sql, value):
        self.cursor.execute(sql, value)
        batch_data = self.cursor.fetchall()
        return batch_data

    def check_table_info(self, table_name):
        sql = f'''PRAGMA table_info({table_name})'''
        self.cursor.execute(sql)
        info = self.cursor.fetchall()
        col_name = [i[1] for i in info]
        return col_name

    def read_id(self, table_name, key_ls):
        sign = ','.join(['?'] * len(key_ls))
        sql = f'''select id from {table_name} where grp in ({sign}) order by perm_order'''
        # sql = f'''select id from {table_name} where grp in ({sign})'''
        self.cursor.execute(sql, key_ls)
        id_column = self.cursor.fetchall()
        # 将"id"列数据转化为列表
        id_list = [row[0] for row in id_column]
        return id_list

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cursor.close()
        self.conn.commit()
        self.conn.close()


def get_train_org_db(train_org_path, train_size):
    # 从train.db中取子集并过滤脏数据 存到org.db
    db_path = mc.config['db_path']
    if os.path.exists(train_org_path):
        print(f"{train_org_path} already exists ")
        return

    with SQLiteHelper(db_path) as db_helper:
        key_ls = mc.config["key_ls"]
        sign = ','.join(['?'] * len(key_ls))
        query_sql = f"""select fingerprint, smiles_canonical, mf, grp, perm_order 
                    from compounds where grp in ({sign})"""
        src_data = db_helper.read(query_sql, size=train_size, key_ls=key_ls)

    with SQLiteHelper(train_org_path) as db_helper:
        # 创建表
        db_helper.create_schema()
        table_name = mc.config['table_name']

        insert_sql = f'insert into {table_name} (id, fp, smiles, mf, mf_vec, grp, perm_order) values(?, ?, ?, ?, ?, ?, ?)'

        for idx, mol in enumerate(src_data):
            fp_bytes, smiles, mf, fold_x, perm_order = mol
            mf_counter = pickle.loads(mf)
            # if not is_valid_molecule(smiles):
            #     continue
            # if len(mf_counter) == 0:
            #     continue
            # ===========处理FP=============================
            # bytes[] --> np
            # fp8925 = fp_unpack(fp_bytes)
            # fp8925 = np.unpackbits(np.frombuffer(fp_bytes, dtype=np.uint8))[:8925]
            # fp3609 = fp8925[map_ls]
            # fp3609 = fp8925_to3609_item(fp8925, map_ls)
            # fp = np.packbits(fp3609)  # 压缩成字符数组
            # ===========处理MF=============================
            # mf:counter --> np.int32 (10,)
            mf_vec = counter_to_vec(mf_counter)
            # "fp":bytes, "mf_vec":np.int32, "smiles":str

            db_helper.execute(insert_sql, (idx, fp_bytes, smiles, mf, mf_vec, fold_x, perm_order))
            if idx % 999 == 0:
                print(f"extract src data {idx}")
    print("already generate org.db")
    print()


def get_k_fold_train_val_set(train_org_path, cv_fold):
    table_name = mc.config['table_name']
    val_key = mc.config['key_ls'][cv_fold]
    train_key = [k for k in mc.config['key_ls'] if k != val_key]

    with SQLiteHelper(train_org_path) as db_helper:
        train_id_ls = db_helper.read_id(table_name, train_key)
        val_id_ls = db_helper.read_id(table_name, [val_key])
        print(len(train_id_ls))  # 110,9435
        print(len(val_id_ls))    # 12,2749

    train_set = FoldDataset(train_org_path, table_name, cv_fold, tv_map=train_id_ls)
    val_set = FoldDataset(train_org_path, table_name, cv_fold, tv_map=val_id_ls)
    train_set.close_conn()
    val_set.close_conn()
    # print(train_set.__getitem__(0))
    return train_set, val_set


# nohup python3 test.py &
def construct_train_val_set(train_org_path, cv_fold, size=1000):
    # 1、train.db取子集src.db,处理+清洗 生成org.db
    get_train_org_db(train_org_path, train_size=size)

    # 2、org.db mf2vec
    # data_2vec(train_org_path)

    # 3、add error to fp, 生成sim-fp, 存到db中
    # cv_fold
    # gen_sim_fp(train_org_path, cv_fold)

    # # 4、划分数据集
    train_set, val_set = get_k_fold_train_val_set(train_org_path, cv_fold)
    # d = train_set.__getitem__(0)
    return train_set, val_set


def check_db(db_path):
    with SQLiteHelper(db_path) as db_helper:
        # query_sql = """select sim_fp_fold0 from org_data"""
        # fp_key = mc.config['sim_fp_ls'][0]
        # query_sql = f'select fp, mf_vec,mf_str,smiles_str from org_data'
        cols = db_helper.check_table_info("compounds")
        query_sql = f'select * from compounds'
        # query_sql = f'select * from org_data'
        data = db_helper.read(query_sql, size=1000)
        # col_ls = db_helper.check_table_info(table_name=train_config.table_name)
        # print(col_ls)
        # fp = np.frombuffer(data[0][2])
    print("HHH")


def t():
    # 一维
    fp = np.random.randint(2, size=(3609,))
    fp_b = np.packbits(fp)
    fp_r = np.unpackbits(np.frombuffer(fp_b, dtype=np.uint8))[:3609]

    # 二维
    fp = np.random.randint(2, size=(2, 3609))
    fp_b = np.packbits(fp, axis=1)
    fp_r = np.unpackbits(np.frombuffer(fp_b, dtype=np.uint8).reshape(2, -1), axis=1)[:, :3609]
    print("HHH")


if __name__ == "__main__":
    # t()
    size = 123
    batch_size = 256
    train_org_path = f"/home/sf123/MS_DATA/org_{size}_order.db"
    # check_db(train_org_path)
    cv_fold = 0
    train_set, val_set = construct_train_val_set(train_org_path, cv_fold=cv_fold, size=size)
    # # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    # sampler = get_sampler(cv_fold, "pt")
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True,
    #                         collate_fn=lambda batch: my_collate_fn(batch, sampler), num_workers=4)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
    # for idx, batch in enumerate(val_loader):
    #     print(batch)
    #     fp_ = batch[0]
    #     fp = sampler.sample_(fp_)
    #     print(idx)
    # fp1 = next(iter(val_loader))[0]
    # fp2 = next(iter(val_loader))[0]
    # print(torch.equal(fp1, fp2))
    # collate_fn_with_args = partial(my_collate_fn, test_dict=test_dict)
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=lambda batch: my_collate_fn(batch, test_dict), num_workers=4)

# fp train: 1116 * 8 = 8928 ?8925

# combined_0824_v44.db
# id  fingerprint  fingerprint_degraded  smiles_generic  smiles_canonical  inchikey  inchikey1  mol  mf  mf_text  source  grp  perm_order

# <class 'list'>
# baseline_sirius_top10_coverage_1604438836.pkl
# nn        rank    inchikey        score       smiles_in       smiles      smiles_generic      smiles_canonical        mol         inchikey1       mf      fingerprint(1, 8925)
# nn_ref            inchikey_ref    score_ref   smiles_in_ref   smiles_ref  smiles_generic_ref  smiles_canonical_ref    mol_ref     inchikey1_ref   mf_ref  fingerprint_ref(3609,)
#                                   match_score                                                                                                             fingerprint_ref_true(1, 8925)
# E:\dataset\val\results\results

# def gen_sim_fp(train_org_path, cv_fold):
#     table_name = mc.config['table_name']
#     with SQLiteHelper(train_org_path) as db_helper:
#         exists_col = db_helper.check_table_info(table_name)
#         sim_fp_key = mc.config['sim_fp_ls'][cv_fold]
#         # sim_fp_key = "sim_fp_fold0_tf"
#         print(f"sim_fp_key: {sim_fp_key}")
#
#         # 如果sim_fp_foldi列已经生成, 则直接返回
#         if sim_fp_key in exists_col:
#             return
#         # 向table添加新列
#         alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {sim_fp_key} BLOB"
#         db_helper.execute(alter_sql)
#
#         map_path = mc.config['fp_map']
#         map_ls = read_map(map_path)
#         # 拿到用于生成sim-fp的测试集
#         sampler = get_sampler(cv_fold=0, option="pt")
#
#         # 遍历处理org.db的数据, 每次取batch_size个
#         offset = 0
#         batch_size = 2048
#         query_sql = f"SELECT id, fp FROM {table_name} LIMIT ? OFFSET ?"
#         cnt = 0
#         while True:
#             # 每次读取batch_size条数据
#             batch_data = db_helper.read_batch(query_sql, batch_size, offset)
#             if not batch_data:
#                 break
#             # ================
#             id_ls = [d[0] for d in batch_data]
#             fp_vec_ = fp_pipeline_unpack([d[1] for d in batch_data])  # np: (batch_size, 8925)
#             struct_fp = fp_vec_[:, map_ls].astype(np.float32)  # np: (batch_size, 3609)
#
#             # add error to fp
#             sim_fp = add_error(struct_fp, sampler).astype(np.float32)
#
#             for i in range(len(batch_data)):
#                 update_sql = f"UPDATE {table_name} SET {sim_fp_key} = ? WHERE id = ?"
#                 # sim_fp_foldi: np.float32
#                 db_helper.execute(update_sql, (sim_fp[i].tobytes(), id_ls[i]))
#
#             offset += batch_size
#             cnt += 1
#             print(f"convert {cnt} * {batch_size} to sim-fp")
#             # ====================
#     print("convert all struct_fp to sim_fp")


# def data_2vec(train_org_path):
#     # 把分子式mf转为向量（10，）
#     table_name = mc.config['table_name']
#     with SQLiteHelper(train_org_path) as db_helper:
#         # table添加新列"mf_vec"
#         exists_col = db_helper.check_table_info(table_name)
#         new_col_ls = ["mf_vec"]
#         # 如果新列已经生成, 则直接返回
#         is_subset = set(new_col_ls).issubset(set(exists_col))
#         if is_subset:
#             return
#         for col in new_col_ls:
#             if col in exists_col:
#                 continue
#             alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {col} BLOB"
#             db_helper.execute(alter_sql)
#
#         # 遍历处理org.db的数据, 每次取batch_size个
#         offset = 0
#         batch_size = 2048
#         query_sql = f"SELECT * FROM {table_name} LIMIT ? OFFSET ?"
#         cnt = 0
#         while True:
#             # 每次读取batch_size条数据
#             batch_data = db_helper.read_batch(query_sql, batch_size, offset)
#             if not batch_data:
#                 break
#
#             mf_vec_ls = []
#             id_ls = []
#             # mf, smiles转vec
#             for idx, mol in enumerate(batch_data):
#                 # id, fold, fp, smiles_str, mf_str = mol[:5]
#                 id, fold, _, smiles_str, mf_str = mol[:5]
#                 # mf转vec
#                 mf_vec = mf2vec(mf_str)  # (10, )
#                 mf_vec_ls.append(mf_vec)
#
#                 id_ls.append(id)
#
#             # 插入到db中
#             for i in range(len(batch_data)):
#                 # update_sql = f"UPDATE {table_name} SET sim_fp = ?, mf_vec = ?, smiles_gt = ?, smiles_lab = ? WHERE id = ?"
#                 update_sql = f"UPDATE {table_name} SET mf_vec = ? WHERE id = ?"
#                 # mf_vec:int32
#                 # db_helper.execute(update_sql, (sim_fp[i], mf_vec_ls[i], smiles_gt_vec[i], smiles_lab_ls[i], id_ls[i]))
#
#                 db_helper.execute(update_sql, (mf_vec_ls[i], id_ls[i]))
#
#             offset += batch_size
#             cnt += 1
#             print(f"convert {cnt} * {batch_size} to vec")
#
#     print("convert all data to vec")

# def my_collate_fn(batch, sampler):
#     # fp, mf, smiles_gt, smiles_lab, smiles_str
#     struct_fp = torch.stack([data[0] for data in batch])
#     # sim_fp, _ = sampler.sample_(struct_fp)
#
#     mf = torch.stack([data[1] for data in batch])
#     smiles_gt = torch.stack([data[2] for data in batch])
#     smiles_lab = torch.stack([data[3] for data in batch])
#     smiles_str = [data[4] for data in batch]
#
#     return struct_fp, mf, smiles_gt, smiles_lab, smiles_str