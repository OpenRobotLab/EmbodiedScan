# ScanNet Instructions

To acquire the access to ScanNet dataset, Please refer to the [ScanNet project page](https://github.com/ScanNet/ScanNet) and follow the instructions there. You will get a `download-scannet.py` script after your request for the ScanNet dataset is approved. Note that only a subset of ScanNet is needed. Once you get `download-scannet.py`, please use the commands below to download the portion of ScanNet that is necessary for ScanRefer:

```shell
python2 download-scannet.py -o data/scannet --type _vh_clean_2.ply
python2 download-scannet.py -o data/scannet --type .aggregation.json
python2 download-scannet.py -o data/scannet --type _vh_clean_2.0.010000.segs.json
python2 download-scannet.py -o data/scannet --type .txt
```

Roughly 10.6GB free space is needed on your disk.
