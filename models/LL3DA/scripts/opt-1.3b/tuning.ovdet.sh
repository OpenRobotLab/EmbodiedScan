export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --pretrained_weights ./ckpts/opt-1.3b/ll3da-generalist/checkpoint.pth \
    --warm_lr_epochs 1 \
    --dataset unified_ovdet_nr3d,unified_ovdet_scanrefer \
    --vocab facebook/opt-1.3b \
    --qformer_vocab bert-base-embedding \
    --checkpoint_dir ./ckpts/opt-1.3b/ll3da-ovdet \
    --max_epoch 32 \
    --dist_url tcp://localhost:123 \
    --eval_every_iteration 4000 \
    --start_eval_after -1 \
    --save_every 10000 \
    --criterion 'CiDEr@0.5' \
    --freeze_detector --freeze_llm \
    --batchsize_per_gpu 4 --ngpus 8 --base_lr 1e-4 --final_lr 1e-6 \
    --max_des_len 256 \
    --max_prompt 1
