import argparse
import numpy as np
import os
import struct
from PIL import Image
import warnings
import os

warnings.filterwarnings('ignore') # 屏蔽nan与min_depth比较时产生的警告

camnum = 12
fB = 32504
min_depth_percentile = 2
max_depth_percentile = 98
depthmapsdir = 'D:\\nerf_data\\360garden_4\\depth_maps_bin\\'
outputdir = 'D:\\nerf_data\\360garden_4\\depth_maps\\'

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def bin2depth(i, depth_map, depthdir):
    # depth_map = '0.png.geometric.bin'
    # print(depthdir)
    # if min_depth_percentile > max_depth_percentile:
    #     raise ValueError("min_depth_percentile should be less than or equal "
    #                      "to the max_depth_perceintile.")

    # Read depth and normal maps corresponding to the same image.
    if not os.path.exists(depth_map):
        raise FileNotFoundError("file not found: {}".format(depth_map))

    depth_map = read_array(depth_map)

    min_depth, max_depth = np.percentile(depth_map[depth_map>0], [min_depth_percentile, max_depth_percentile])
    depth_map[depth_map <= 0] = np.nan # 把0和负数都设置为nan，防止被min_depth取代
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth

    maxdisp = fB / min_depth
    mindisp = fB / max_depth
    depth_map = (fB/depth_map - mindisp) * 255 / (maxdisp - mindisp)
    depth_map = np.nan_to_num(depth_map) # nan全都变为0
    depth_map = depth_map.astype(int)

    #image = Image.fromarray(depth_map).convert('L')
    # image = image.resize((1920, 1080), Image.ANTIALIAS) # 保证resize为1920*1080
    #mage.save(depthdir + str(i) + '.png')

files = os.listdir(depthmapsdir)
for file_name in files:
    if "geometric" in file_name:
        file_path = os.path.join(depthmapsdir, file_name)
        bin2depth(file_name,file_path,outputdir)
