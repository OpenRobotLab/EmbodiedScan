export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=0,1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --pretrained_weights /mnt/petrelfs/linjingli/mmscan_modelzoo-main/llmzoo/LL3DA/pretrained/ll-gernalist/ll3da-opt-1.3b.pth \
    --warm_lr_epochs 0 \
    --dataset unified_scanqa \
    --vocab facebook/opt-1.3b \
    --qformer_vocab bert-base-embedding \
    --checkpoint_dir ./ckpts/opt-1.3b/ll3da-scanqa-tuned \
    --max_epoch 24 \
    --dist_url tcp://localhost:333 \
    --eval_every_iteration 4000 \
    --start_eval_after -1 \
    --save_every 10000 \
    --criterion 'CiDEr' \
    --freeze_detector --freeze_llm \
    --batchsize_per_gpu 8 --ngpus 2 --base_lr 1e-6 --final_lr 1e-6 \
    --max_des_len 224 \
    --max_prompt 1 --use_beam_search
