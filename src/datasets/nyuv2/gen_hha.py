import getopt
import sys
import os
from PIL import Image
import numpy as np
import h5py

from rgbd_util import getHHA


def save_hha(f_h5, camera_mats, dir_hha_out):
    if not os.path.isdir(dir_hha_out):
        os.makedirs(dir_hha_out)

    # 读取train.txt test.txt hha分别存放
    train_txt = os.path.join(dir_hha_out, 'train.txt')
    test_txt = os.path.join(dir_hha_out, 'test.txt')
    all_images = 1449
    img_name = [0] * all_images
    with open(train_txt) as f:
        for line in f:
            img_name[int(line)-1] = 'train'
    with open(test_txt) as f:
        for line in f:
            img_name[int(line)-1] = 'test'


    depths = np.array(f_h5["depths"])
    for i, depth in enumerate(depths):
        depth = depth.transpose((1, 0))
        # print('depth_min', depth.min())
        # print('depth_max', depth.max())
        # print(camera_mats[i])
        hha = getHHA(camera_mats[i], depth, depth)
        # print(hha.min(), hha.max())
        hha_img = Image.fromarray(hha)
        hha_path = os.path.join(dir_hha_out, 'hha')
        if img_name[i] == 'train':
            hha_path = os.path.join(dir_hha_out, 'train')
        else:
            hha_path = os.path.join(dir_hha_out, 'test')
        hha_path = os.path.join(dir_hha_out, "%06d.png" % (i+1))
        hha_img.save(hha_path, 'PNG', optimize=True)

        print('hha', i)
        # break

def get_nyu_cam_mats(path_nyu_cam_mats):
    mats = []
    with open(path_nyu_cam_mats, 'r') as f_cam:
        lines = f_cam.readlines()
        for line_id in range(0, len(lines), 4):
            # this camera paras from sun_rgbd's nyu_v2
            # paras = "518.857901 0.000000 284.582449 0.000000 519.469611 208.736166 0.000000 0.000000 1.000000"
            # numbers = np.array([paras.split(' ')[:9]]).astype(float)
            # mat = np.reshape(numbers, [3, 3], 'C')
            mat = np.zeros([3, 3], dtype=np.float)
            for i in range(3):
                line = lines[line_id + i]
                eles = line.split(' ')
                for j in range(len(eles)):
                    if i == 2:
                        mat[i][j] = float(eles[j])
                    else:
                        mat[i][j] = float(eles[j]) * 1000
            mats.append(mat)
    return mats


def main(dir_meta, dir_out):
    path_nyu_depth_v2_labeled = os.path.join(dir_meta, "nyu_depth_v2_labeled.mat")
    f = h5py.File(path_nyu_depth_v2_labeled)
    print(f.keys())


    pth_cam_mats = os.path.join(dir_meta, "camera_rotations_NYU.txt")
    # dir_sub_out = os.path.join(dir_out, 'hha')
    dir_sub_out = dir_out
    save_hha(f, get_nyu_cam_mats(pth_cam_mats), dir_sub_out)


if __name__ == '__main__':
    input_dir = ""
    output_dir = ""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["input_meta_dir=", "output_dir="])
    except getopt.GetoptError:
        print('gen_nyu.py -i <input_meta_dir> -o <output_dir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('gen_nyu.py -i <input_meta_dir> -o <output_dir>')
            sys.exit()
        elif opt in ("-i", "--input_meta_dir"):
            input_dir = arg
        elif opt in ("-o", "--output_dir"):
            output_dir = arg

    main(input_dir, output_dir)
