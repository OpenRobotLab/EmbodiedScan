## 3D Visual Grounding Models

These are 3D visual grounding models adapted for the mmscan-devkit. Currently, two models have been released: EmbodiedScan and ScanRefer.

### Scanrefer

1. Follow the [Scanrefer](https://github.com/daveredrum/ScanRefer/blob/master/README.md) to setup the Env. For data preparation, you need not load the datasets, only need to download the [preprocessed GLoVE embeddings](https://kaldir.vc.in.tum.de/glove.p) (~990MB) and put them under `data/`

2. Install MMScan API.

3. Overwrite the `lib/config.py/CONF.PATH.OUTPUT` to your desired output directory.

4. Run the following command to train Scanrefer (one GPU):

   ```bash
   python -u scripts/train.py --use_color --epoch {10/25/50}
   ```

5. Run the following command to evaluate Scanrefer (one GPU):

   ```bash
   python -u scripts/train.py --use_color --eval_only --use_checkpoint "path/to/pth"
   ```
#### ckpts & Logs

| Epoch |   gtop-1 @ 0.25/0.50  |                           Config                           |                                                                                                                                                                 Download                                                                                                                                                                 |
| :-------:   | :---------: | :--------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| 50 |  4.74 / 2.52    |    [config](https://drive.google.com/file/d/1iJtsjt4K8qhNikY8UmIfiQy1CzIaSgyU/view?usp=drive_link)    |             [model](https://drive.google.com/file/d/1C0-AJweXEc-cHTe9tLJ3Shgqyd44tXqY/view?usp=drive_link) \| [log](https://drive.google.com/file/d/1ENOS2FE7fkLPWjIf9J76VgiPrn6dGKvi/view?usp=drive_link)  
### EmbodiedScan

1. Follow the [EmbodiedScan](https://github.com/OpenRobotLab/EmbodiedScan/blob/main/README.md) to setup the Env. Download the [Multi-View 3D Detection model's weights](https://download.openmmlab.com/mim-example/embodiedscan/mv-3ddet.pth) and change the "load_from" path in the config file under `configs/grounding` to the path where the weights are saved.

2. Install MMScan API.

3. Run the following command to train EmbodiedScan (multiple GPU):

   ```bash
   # Single GPU training
   python tools/train.py configs/grounding/pcd_4xb24_mmscan_vg_num256.py --work-dir=path/to/save

   # Multiple GPU training
   python tools/train.py configs/grounding/pcd_4xb24_mmscan_vg_num256.py --work-dir=path/to/save --launcher="pytorch"
   ```

4. Run the following command to evaluate EmbodiedScan (multiple GPU):

   ```bash
   # Single GPU testing
   python tools/test.py configs/grounding/pcd_4xb24_mmscan_vg_num256.py path/to/load_pth

   # Multiple GPU testing
   python tools/test.py configs/grounding/pcd_4xb24_mmscan_vg_num256.py path/to/load_pth --launcher="pytorch"
   ```
#### ckpts & Logs

| Input-modality  | Load pretrain | epoch |  gtop-1 @ 0.25/0.50  |                           Config                           |                                                                                                                                                                 Download                                                                                                                                                                 |
| :-------:  | :----: | :----: | :---------: | :--------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Point cloud   |  True  |  12 |  19.66 / 8.82     |    [config](https://github.com/rbler1234/EmbodiedScan/blob/mmscan-devkit/models/EmbodiedScan/configs/grounding/pcd_4xb24_mmscan_vg_num256.py)    |             [model](https://drive.google.com/file/d/1F6cHY6-JVzAk6xg5s61aTT-vD-eu_4DD/view?usp=drive_link) \| [log](https://drive.google.com/file/d/1Ua_-Z2G3g0CthbeBkrR1a7_sqg_Spd9s/view?usp=drive_link)  

## 3D Question Answering Models

These are 3D question answering models adapted for the mmscan-devkit. Currently, two models have been released: LL3DA and LEO.

### LL3DA

1. Follow the [LL3DA](https://github.com/Open3DA/LL3DA/blob/main/README.md) to setup the Env. For data preparation, you need not load the datasets, only need to:

   (1) download the [release pre-trained weights.](https://huggingface.co/CH3COOK/LL3DA-weight-release/blob/main/ll3da-opt-1.3b.pth) and put them under `./pretrained`

   (2) Download the [pre-processed BERT embedding weights](https://huggingface.co/CH3COOK/bert-base-embedding/tree/main) and store them under the `./bert-base-embedding` folder

2. Install MMScan API.

3. Edit the config under `./scripts/opt-1.3b/eval.mmscanqa.sh` and `./scripts/opt-1.3b/tuning.mmscanqa.sh`

4. Run the following command to train LL3DA (4 GPU):

   ```bash
   bash scripts/opt-1.3b/tuning.mmscanqa.sh
   ```

5. Run the following command to evaluate LL3DA (4 GPU):

   ```bash
   bash scripts/opt-1.3b/eval.mmscanqa.sh
   ```

   Optinal: You can use the GPT evaluator by this after getting the result.
   'qa_pred_gt_val.json' will be generated under the checkpoint folder after evaluation and the tmp_path is used for temporarily storing.

   ```bash
   python eval_utils/evaluate_gpt.py --file path/to/qa_pred_gt_val.json
   --tmp_path path/to/tmp  --api_key your_api_key --eval_size -1
   --nproc 4
   ```
#### ckpts & Logs

| Detector  | Captioner | Iters |  GPT score overall  |                                                                                                                                                                       Download                                                                                                                                                                 |
| :-------:  | :----: | :----: | :---------: |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| detector_Vote2Cap_DETR   |  ll3da  |  100k |  45.7     |             [model](https://drive.google.com/file/d/1mcWNHdfrhdbtySBtmG-QRH1Y1y5U3PDQ/view?usp=drive_link) \| [log](https://drive.google.com/file/d/1VHpcnO0QmAvMa0HuZa83TEjU6AiFrP42/view?usp=drive_link)             |



### LEO

1. Follow the [LEO](https://github.com/embodied-generalist/embodied-generalist/blob/main/README.md) to setup the Env. For data preparation, you need not load the datasets, only need to:

   (1) Download [Vicuna-7B](https://huggingface.co/huangjy-pku/vicuna-7b/tree/main) and update cfg_path in configs/llm/\*.yaml

   (2) Download the [sft_noact.pth](https://huggingface.co/datasets/huangjy-pku/LEO_data/tree/main) and store it under the `./weights` folder

2. Install MMScan API.

3. Edit the config under `scripts/train_tuning_mmscan.sh` and `scripts/test_tuning_mmscan.sh`

4. Run the following command to train LEO (4 GPU):

   ```bash
   bash scripts/train_tuning_mmscan.sh
   ```

5. Run the following command to evaluate LEO (4 GPU):

   ```bash
   bash scripts/test_tuning_mmscan.sh
   ```

   Optinal: You can use the GPT evaluator by this after getting the result.
   'test_embodied_scan_l_complete.json' will be generated under the checkpoint folder after evaluation and the tmp_path is used for temporarily storing.

   ```bash
   python evaluator/GPT_eval.py --file path/to/test_embodied_scan_l_complete.json
   --tmp_path path/to/tmp  --api_key your_api_key --eval_size -1
   --nproc 4
   ```
#### ckpts & Logs

| LLM  | Vision2d/3d | epoch |  GPT score overall  |                           Config                           |                                                                                                                                                                 Download                                                                                                                                                                 |
| :-------:  | :----: | :----: | :---------: | :--------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| vicuna7b   |  convnext / ose3d_pointnetpp  |  1 |  54.6     |    [config](https://drive.google.com/file/d/1CJccZd4TOaT_JdHj073UKwdA5PWUDtja/view?usp=drive_link)    |             [model](https://drive.google.com/drive/folders/1HZ38LwRe-1Q_VxlWy8vqvImFjtQ_b9iA?usp=drive_link)              |
