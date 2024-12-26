export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
# unified_embodied_scan_caption
#/mnt/hwfile/OpenRobotLab/yangshuai1/ll3da/ckpts/fine_tune_full/checkpoint_140k.pth
python main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --test_ckpt  /mnt/petrelfs/linjingli/mmscan_modelzoo-main/llmzoo/LL3DA/ckpts/opt-1.3b/train_qa_7_31/checkpoint_100k.pth \
    --dataset unified_embodied_scan_qa \
    --vocab facebook/opt-1.3b \
    --qformer_vocab bert-base-embedding \
    --checkpoint_dir ckpts/opt-1.3b/test_7_31 \
    --dist_url tcp://localhost:12345 \
    --criterion 'CiDEr' \
    --freeze_detector --freeze_llm \
    --batchsize_per_gpu 12 --ngpus 4 \
    --max_des_len 512 \
    --max_prompt 1 \
    --use_beam_search \
    --test_only

# python main.py \
#     --use_color --use_normal \
#     --detector detector_Vote2Cap_DETR \
#     --captioner ll3da \
#     --checkpoint_dir ./ckpts/opt-1.3b/ll3da-generalist \
#     --test_ckpt ./ckpts/opt-1.3b/ll3da-generalist/checkpoint.pth \
#     --dataset unified_3dllm_scene_description,unified_3dllm_embodied_dialogue,unified_3dllm_embodied_planning,unified_scanqa,unified_densecap_nr3d,unified_densecap_scanrefer \
#     --vocab facebook/opt-1.3b \
#     --qformer_vocab bert-base-embedding \
#     --dist_url tcp://localhost:12345 \
#     --criterion 'CiDEr' \
#     --freeze_detector --freeze_llm \
#     --batchsize_per_gpu 4 --ngpus 8 \
#     --max_des_len 512 \
#     --max_prompt 1 \
#     --use_beam_search \
#     --test_only
