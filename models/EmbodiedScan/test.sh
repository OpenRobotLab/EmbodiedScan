python tools/test.py configs/grounding/pcd_vg_1030.py /mnt/petrelfs/linjingli/tmp/code/MMScan-code/VG/benchmark/EmbodiedScan/exps/MMScan-VG-1030/epoch_12.pth --work-dir exps/MMScan-VG-1030 --launcher="slurm"
# GPUS=4
# CONFIG=configs/grounding/pcd_vg_1030.py
# WORK_DIR=exps/MMScan-VG-1030
# PORT=`expr $RANDOM % 4000 + 25000`

# python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" --cfg-options env_cfg.dist_cfg.port=${PORT}
