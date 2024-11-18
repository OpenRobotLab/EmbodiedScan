python -m torch.distributed.launch --nproc_per_node=4 tools/train.py configs/grounding/pcd_vg_1030.py --work-dir exps/MMScan-VG-1030 --launcher="pytorch"
