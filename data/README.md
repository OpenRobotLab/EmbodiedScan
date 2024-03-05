### Prepare EmbodiedScan Data

Given the licenses of respective raw datasets, we recommend users download the raw data from their official websites and then organize them following the below guide.
Detailed steps are shown as follows.

1. Download ScanNet v2 data [HERE](https://github.com/ScanNet/ScanNet). Link or move the folder to this level of directory.

2. Download 3RScan data [HERE](https://github.com/WaldJohannaU/3RScan). Link or move the folder to this level of directory.

3. Download Matterport3D data [HERE](https://github.com/niessner/Matterport). Link or move the folder to this level of directory.

4. Download EmbodiedScan data and extract it here. Currently, please fill in the [form](https://docs.google.com/forms/d/e/1FAIpQLScUXEDTksGiqHZp31j7Zp7zlCNV7p_08uViwP_Nbzfn3g6hhw/viewform?usp=sf_link), and we will reply with the data download link.

The directory structure should be as below.

```
data
├── scannet
│   ├── scans
│   │   ├── <scene_id>
│   │   ├── ...
├── 3rscan
│   ├── <scene_id>
│   ├── ...
├── matterport3d
│   ├── <scene_id>
│   ├── ...
├── embodiedscan_occupancy
├── embodiedscan_infos_train.pkl
├── embodiedscan_infos_val.pkl
├── embodiedscan_infos_train_full_vg.json
├── embodiedscan_infos_val_full_vg.json
├── embodiedscan_infos_train_mini_vg.json
├── embodiedscan_infos_val_mini_vg.json
```

5. Enter the project root directory, extract images by running

```bash
python embodiedscan/converter/generate_image_scannet.py --dataset_folder data/scannet/
# generate_image_scannet.py can be very slow because it extracts images from .sens files. Add --fast to generate only images used by embodiedscan.
python embodiedscan/converter/generate_image_3rscan.py --dataset_folder data/3rscan/
```

The directory structure should be as below after that

```
data
├── scannet
│   ├── scans
│   │   ├── <scene_id>
│   │   ├── ...
│   ├── posed_images
│   │   ├── <scene_id>
│   │   |   ├── *.jpg
│   │   |   ├── *.png
│   │   ├── ...
├── 3rscan
│   ├── <scene_id>
│   │   ├── sequence
│   │   |   ├── *.color.jpg
│   │   |   ├── *.depth.pgm
│   ├── ...
├── matterport3d
│   ├── <scene_id>
│   ├── ...
├── embodiedscan_occupancy
├── embodiedscan_infos_train_full.pkl
├── embodiedscan_infos_val_full.pkl
├── embodiedscan_infos_train_full_vg.json
├── embodiedscan_infos_val_full_vg.json
├── embodiedscan_infos_train_mini_vg.json
├── embodiedscan_infos_val_mini_vg.json
```

6. Also extract EmbodiedScan occupancy annotations here by running

```bash
python embodiedscan/converter/extract_occupancy_ann.py --src data/embodiedscan_occupancy --dst data
```
