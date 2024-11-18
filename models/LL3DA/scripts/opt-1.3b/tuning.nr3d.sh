export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=4,5
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --pretrained_weights ./ckpts/opt-1.3b/ll3da-generalist/checkpoint.pth \
    --warm_lr_epochs 0 \
    --dataset unified_densecap_nr3d \
    --vocab facebook/opt-1.3b \
    --qformer_vocab bert-base-embedding \
    --checkpoint_dir ./ckpts/opt-1.3b/ll3da-nr3d-tuned \
    --max_epoch 32 \
    --dist_url tcp://localhost:111 \
    --eval_every_iteration 4000 \
    --start_eval_after -1 \
    --save_every 10000 \
    --criterion 'CiDEr@0.5' \
    --freeze_detector --freeze_llm \
    --batchsize_per_gpu 8 --ngpus 2 --base_lr 1e-6 --final_lr 1e-6 \
    --max_des_len 256 \
    --max_prompt 1 --use_beam_search
