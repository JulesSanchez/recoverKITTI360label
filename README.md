# recoverKITTI360label

Python partial re-implementation of accumuLaser in python from the KITTI360 devkits to recover label of individual pointclouds from aggregated pointclouds. I am not affiliated with the original authors.
You can find the original implementation at https://github.com/autonomousvision/kitti360Scripts/tree/master/kitti360scripts/devkits/accumuLaser

# Limitations

It can only work on VelodyneData, it would require to implement loadSickData to support sick data.

As the aim of this work was to recover the label of the indivdual frame inside KITTI360 and not to recreate aggregated pointclouds, the "addQdToMd" function doesn't use "sparsifyData" in order to keep most of the points.

In pratice, I changed some hyperparameters forme the one in the original code, most notably line 313, I set -1 as the blind_spot_angle.

# Usage 

main.py can be used to create accumulated pointclouds, following the original goal of accumuLaser
recoverLabels.py can be used to generate label files for each individual frame. It is based on a nearest neighbor search on the current previous and next accumulated static and dynamic pointclouds that correspond to this frame numer.

![visualization_seq3](https://user-images.githubusercontent.com/94064424/146905067-fd462f84-40ac-4c54-91a1-aa543d324dc2.gif)
An example where the raw points were visualized with the recovered labels using a 2D spherical projection.

# KITTI360

Details and download are available at: www.cvlibs.net/datasets/kitti-360

@INPROCEEDINGS{Xie2016CVPR,
author = {Jun Xie and Martin Kiefel and Ming-Ting Sun and Andreas Geiger},
title = {Semantic Instance Annotation of Street Scenes by 3D to 2D Label Transfer},
booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2016}
}
