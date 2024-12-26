#!/usr/bin/env bash

set -x

CKPT_PATH=/mnt/petrelfs/lvruiyuan/repos/EmbodiedScan/work_dirs
PARTITION=mozi-S1
JOB_NAME=mv-grounding-challenge-benchmark
TASK=mv-grounding-challenge-benchmark
CONFIG=configs/grounding/mv-grounding_8xb12_embodiedscan-vg-9dof.py
WORK_DIR=${CKPT_PATH}/${TASK}
CKPT=${CKPT_PATH}/${TASK}/latest.pth
CPUS_PER_TASK=16
GPUS=8
GPUS_PER_NODE=8
PORT=29320

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
export NCCL_IB_DISABLE=1; export NCCL_P2P_DISABLE=1; \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=reserved \
    python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" --cfg-options env_cfg.dist_cfg.port=${PORT} --resume
