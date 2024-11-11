#!/bin/bash

# 检查 environment.yml 文件是否存在
if [ ! -f environment.yml ]; then
  echo "Error: environment.yml file not found!"
  exit 1
fi

# 创建 Conda 环境
conda env create -f environment.yml

# 检查环境创建是否成功
if [ $? -eq 0 ]; then
  echo "Conda environment created successfully."
else
  echo "Failed to create Conda environment."
  exit 1
fi