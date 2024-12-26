export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# change the test_ckpt and checkpoint_dir

python main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner ll3da \
    --checkpoint_dir path/to/record \
    --test_ckpt path/to/ckpt \
    --dataset unified_embodied_scan_qa \
    --vocab facebook/opt-1.3b \
    --qformer_vocab bert-base-embedding \
    --dist_url tcp://localhost:1233 \
    --criterion 'refined_EM' \
    --freeze_detector --freeze_llm \
    --batchsize_per_gpu 8 --ngpus 4 \
    --max_des_len 224 \
    --max_prompt 1 \
    --use_beam_search \
    --test_only
