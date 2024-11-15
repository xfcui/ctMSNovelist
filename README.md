## ctMSNovelist

Paper: How to Train Your Neural Network for Molecular Structure Generation from Mass Spectra?

Authors: Kai Zhao, Yanmin Liu, Longyang Dian, Shiwei Sun\*, Xuefeng Cui\*

Contact:  kaizhao@mail.sdu.edu.cn, xfcui@email.sdu.edu.cn

## Create Environment

```
chmod +x create_env.sh
./create_env.sh
```

Notes: 

1、You need to download [jdk11](https://zenodo.org/records/14168787) to the **fp_management** directory.

2、You need to download [complete_folds_smiles_holdout_me.pkl](https://zenodo.org/records/14168787) and put it in the **MS_DATA** directory.

3、You need to download the [model parameter files](https://zenodo.org/records/14168787) and place them in the **evaluation/weights_end** directory.

## Dataset

You can obtain all the data and model parameter files through the link https://zenodo.org/records/14168787.

## Experiments

**Command line interface**

```
You can reproduce all the results of the experiment with the following command

m2: python evaluate_loop_x.py --expmId m2

m3: python evaluate_loop_x.py --expmId m3

m4: python evaluate_loop_x.py --expmId m4

m5: python evaluate_loop_x.py --expmId m5

m6: python evaluate_loop_x.py --expmId m6

m7: python evaluate_loop_x.py --expmId m7

m7.2: python evaluate_loop_x.py --expmId m7 --beam_width=16

t1: python evaluate_loop_x.py --expmId t1 --model_mode ms2smiles

t2: python evaluate_loop_x.py --expmId t2 --model_mode ms2smiles
```

**All experiments**

|      | **Main experiments** | **Ablation experiments** | **Side experiments** |
| ---- | -------------------- | ------------------------ | -------------------- |
| id   | m7，m7.2             | m2~m7                    | t1，t2               |

**Experimental result table**

|      |              |             |             |        |            | Params     |        | valid SMILES |        | correct MF |         |        |        | retrieval |
| ---- | ------------ | ----------- | ----------- | ------ | ---------- | ---------- | ------ | ------------ | ------ | ---------- | ------- | ------ | ------ | --------- |
| id   | pre-training | fine-tuning | co-training | expmId | beam width | model_mode | mean % | # >0         | mean % | # > 0      | % found | top1   | top5   | top10     |
| m2   | ×            | √           | ×           | m2     | 128        | msnovelist | 31.40% | 99.00%       | 4.70%  | 60.00%     | 12.80%  | 8.70%  | 12.40% | 12.80%    |
| m3   | ×            | √           | √           | m3     | 128        | msnovelist | 45.90% | 99.70%       | 9.20%  | 83.40%     | 19.20%  | 12.00% | 18.10% | 18.70%    |
| m4   | √            | ×           | ×           | m4     | 128        | msnovelist | 78.30% | 100.00%      | 65.70% | 99.10%     | 40.70%  | 21.20% | 33.60% | 36.40%    |
| m5   | √            | ×           | √           | m5     | 128        | msnovelist | 79.90% | 100.00%      | 68.90% | 99.30%     | 40.90%  | 21.90% | 34.10% | 36.70%    |
| m6   | √            | √           | ×           | m6     | 128        | msnovelist | 78.10% | 100.00%      | 66.00% | 99.40%     | 45.30%  | 23.50% | 37.00% | 39.90%    |
| m7   | √            | √           | √           | m7     | 128        | msnovelist | 78.70% | 100.00%      | 60.50% | 99.50%     | 48.80%  | 25.40% | 40.30% | 43.80%    |
| m7.2 | √            | √           | √           | m7     | 16         | msnovelist | 83.52% | 99.85%       | 67.02% | 98.66%     | 37.87%  | 24.14% | 36.09% | 37.66%    |
| t1   | ×            | ×           | ×           | t1     | 128        | ms2smiles  | 80.11% | 100.00%      | 77.54% | 99.56%     | 42.23%  | 23.25% | 35.50% | 37.82%    |
| t2   | √            | √           | √           | t2     | 128        | ms2smiles  | 78.28% | 100.00%      | 73.10% | 99.66%     | 50.61%  | 26.11% | 41.75% | 45.17%    |



## Project structure
- **ctMSNovelist （Project folder）**
  - evaluation
    - topk    (Store test results)
      - m2~m7
      - t1~t2
    - weights_end   (Store the model parameter files)
      - m2~m7
      - t1~t2
  - fp_management
    - jdk-11.0.23
  - MS_DATA （Data folder）  
    - combined_0824_v44.db：  Training and validation data
    - complete_folds_smiles_holdout_me.pkl：  Test data, CANOPUS+GNPS+CASMI
    - csi_fingerid.csv：  Molecular fingerprint mapping  8925-->3609
  - other files

