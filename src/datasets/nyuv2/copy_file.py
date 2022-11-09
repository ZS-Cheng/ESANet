import os
import shutil
import glob

# srcfile 需要复制、移动的文件   
# dstpath 目的地址
def mycopyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        dstpath = os.path.join(dstpath, fname[2:])
        shutil.copy(srcfile, dstpath)          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath))

def copyFile(srcfile,dstpath):                       # 复制函数
    print(srcfile)
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        dstpath = os.path.join(dstpath, fname[2:])
        shutil.copy(srcfile, dstpath)          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath))


def main():
    src_dir = '/content/ESANet/datasets/nyu_v2/hha'
    dst_dir = '/content/ESANet/datasets/nyuv2/hha'                                    # 目的路径记得加斜杠
    src_file_list = glob.glob(src_dir + '/*')                    # glob获得路径下所有文件，可根据需要修改

    # 读取train.txt test.txt hha分别存放
    dir_hha_out = '/content/ESANet/datasets/nyu_v2'
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
    print(img_name[9])
    i = 0
    for srcfile in src_file_list:
        fpath,fname=os.path.split(srcfile)
        index = int(fname[2:6])     # 这个glob把顺序打乱了
        if img_name[index-1] == 'train':
            dst_dir1 = os.path.join(dst_dir, 'train')
        else:
            dst_dir1 = os.path.join(dst_dir, 'test')
        copyFile(srcfile, dst_dir1)                      # 复制文件
        print(fname, dst_dir1)
        i = i + 1


if __name__ == "__main__":
    main()