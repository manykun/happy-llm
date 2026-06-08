#!/bin/bash

set -euo pipefail

# dataset dir 下载到本地目录
dataset_dir="../data"

mkdir -p "${dataset_dir}"

echo "Downloading pretrain dataset to ${dataset_dir}"
modelscope download \
  --dataset ddzhu123/seq-monkey \
  mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 \
  --local_dir "${dataset_dir}"

echo "Extracting pretrain dataset. The archive is 10.6G and expands to roughly 32G."
tar -xvf "${dataset_dir}/mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2" -C "${dataset_dir}"

echo "Downloading SFT dataset to ${dataset_dir}/BelleGroup"
modelscope download \
  --dataset swift/train_3.5M_CN \
  --local_dir "${dataset_dir}/BelleGroup"
