## Set up data
Assuming that you are in the root of the repository
```
mkdir data
ln -s /path/to/kitti_raw_data data/
ln -s /path/to/kitti_scene_flow data/
```

If the machine already has the dataset set up then
```
python setup/setup_dataset_kitti.py --paths_only
```
Otherwise produce data using
```
python setup/setup_dataset_kitti.py --n_thread 8
```