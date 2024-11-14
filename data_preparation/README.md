### Prepare point clouds info files.

Given the licenses of respective raw datasets, we recommend users download the raw data from their official websites and then organize them following the below guide.
Detailed steps are shown as follows.

1. Download ScanNet v2 data [HERE](https://github.com/ScanNet/ScanNet). Link or move the folder to this level of directory.

2. Download 3RScan data [HERE](https://github.com/WaldJohannaU/3RScan). Link or move the folder to this level of directory.

3. Download Matterport3D data [HERE](https://github.com/niessner/Matterport). Link or move the folder to this level of directory.

4. Organize the file structure. Under `mmscan_data/embodiedscan-split/embodiedscan-v1`, the directory structure should be as below,
   You are recommanded to create a soft link to the raw data folder under `mmsan_data/embodiedscan-split/embodiedscan-v1`.

   Additionally, create a `process_pcd` folder in the same directory to store the results. Similarly, we recommend using a symbolic link, as the total file size might be a little large (approximately 21GB)

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
   process_pcd
   embodiedscan_infos_train.pkl
   embodiedscan_infos_val.pkl
   ...
   ```

5. Read raw files and generate processed point cloud files, by running the following scripts.

   ```bash
   python process_all_scan.py --nproc 8
   # If your various file directories do not match the configuration settings, define them using --
   ```
