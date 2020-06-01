import torch
import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt 
from PIL import Image
import matplotlib

from utils import check_dir, build_geo, fbp, ssim_mse_psnr
from datasets import BuildDataSet
from utils import SinoInter, Any2One
from inter_function import newinter


# from torch.utils.data import DataLoader

class InitParser(object):
    def __init__(self):
        ## set parameters
        self.use_cuda = torch.cuda.is_available()
        self.mode = "test"
        self.sparse_view = 60
        self.full_view = 1160
        self.is_shuffle = True if self.mode is "train" else False
        
        self.batch_size= 1
        self.data_length = 224

        # path setting
        if self.use_cuda:
            self.data_root_path = "/mnt/tabgha/users/gy/data/Mayo"
        else:
            self.data_root_path = "V:/users/gy/data/Mayo"
        self.train_folder = {"patients": ["L096","L109","L143","L192","L286","L291","L310","L333", "L506"], "SliceThickness": ["full_3mm"]}
        self.test_folder = {"patients": "L067", "SliceThickness": "full_3mm"}
        self.val_folder = {"patients": "L067", "SliceThickness": "full_3mm"}


def main(args):
    if args.use_cuda:
        print("Using GPU")
    else: 
        print("Using CPU")
    geo_full = build_geo(args.full_view)
    geo_sparse = build_geo(args.sparse_view)
    index = np.random.randint(low=0, high=223)
    index = 185
    # 100
    print("Index:", index)
    datasets = BuildDataSet(args.data_root_path, args.test_folder, geo_full, geo_sparse, None, "test")

    data_length = len(datasets)
    if not data_length == args.data_length:
        print("Args.data_length is wrong!")
        sys.exit(0)
    sample = datasets[index]
    
    image_true = Any2One(sample["image_true"])
    image_full = Any2One(sample["image_full"])
    image_sparse = Any2One(sample["image_sparse"])
    image_inter = Any2One(sample["image_inter"])
    # sinogram_full = sample["sinogram_full"]
    # sinogram_sparse = sample["sinogram_sparse"]
    # sinogram_inter = sample["sinogram_inter"]
    plt.figure()
    plt.subplot(121), plt.xticks([]), plt.yticks([]), plt.imshow(image_full, cmap="gray")
    plt.subplot(122), plt.xticks([]), plt.yticks([]), plt.imshow(image_sparse, cmap="gray")
    plt.show()


    """
    **************************************************
    Image 4-1
    **************************************************
    """
    # sinogram_new = SinoInter(sinogram_inter, geo_full, -3, "sinogram_LineInter", "z")
    # sinogram_new = newinter(sinogram_sparse)
    # image_new = Any2One(fbp(sinogram_new, geo_full))
    # plt.figure()
    # plt.subplot(341), plt.xticks([]), plt.yticks([]), plt.imshow(sinogram_full, cmap="gray"),   plt.title("(a)", y=-0.22)
    # plt.subplot(342), plt.xticks([]), plt.yticks([]), plt.imshow(sinogram_sparse, cmap="gray"), plt.title("(b)", y=-8.5)
    # plt.subplot(343), plt.xticks([]), plt.yticks([]), plt.imshow(sinogram_inter, cmap="gray"),     plt.title("(c)",  y=-0.22)
    # plt.subplot(344), plt.xticks([]), plt.yticks([]), plt.imshow(sinogram_new, cmap="gray"),   plt.title("(d)",  y=-0.22)
    # plt.subplot(345), plt.xticks([]), plt.yticks([]), plt.imshow(image_full, cmap="gray"),   plt.title("(e)", y=-0.22)
    # plt.subplot(346), plt.xticks([]), plt.yticks([]), plt.imshow(image_sparse, cmap="gray"), plt.title("(f)", y=-0.22)
    # plt.subplot(347), plt.xticks([]), plt.yticks([]), plt.imshow(image_inter, cmap="gray"),     plt.title("(g)",  y=-0.22)
    # plt.subplot(348), plt.xticks([]), plt.yticks([]), plt.imshow(image_new, cmap="gray"),   plt.title("(h)",  y=-0.2)
    # plt.subplot(349), plt.xticks([]), plt.yticks([]), plt.imshow(image_full-image_full, cmap="gray"),   plt.title("(i)", y=-0.22)
    # plt.subplot(3,4,10), plt.xticks([]), plt.yticks([]), plt.imshow(image_full-image_sparse, cmap="gray"), plt.title("(j)", y=-0.22)
    # plt.subplot(3,4,11), plt.xticks([]), plt.yticks([]), plt.imshow(image_full-image_inter, cmap="gray"),     plt.title("(k)",  y=-0.22)
    # plt.subplot(3,4,12), plt.xticks([]), plt.yticks([]), plt.imshow(image_full-image_new, cmap="gray"),   plt.title("(l)",  y=-0.22)
    # # plt.savefig("V:/users/gy/MyProject/Resul/Tred/Image/image4-1.png")
    # plt.show()
    """
    **************************************************
    表 4-1
    **************************************************
    """
    # ssim, mse, psnr = ssim_mse_psnr(image_full, image_sparse)
    # print("Sparse:", ssim, mse, psnr)
    # ssim, mse, psnr = ssim_mse_psnr(image_full, image_inter)
    # print("Inter:", ssim, mse, psnr)
    # ssim, mse, psnr = ssim_mse_psnr(image_full, image_new)
    # print("New:", ssim, mse, psnr)

    """
    **************************************************
    Test
    **************************************************
    # """
    # wegs = np.linspace(-10, 10, 200)

    # s = 0
    # m = 0
    # p = 0
    # weg_s = 0
    # weg_p = 0
    # for weg in wegs:
    #     print(".........................Now:{}".format(weg))
    #     sinogram_new = SinoInter(sinogram_inter, geo_full, weg, "sinogram_LineInter", "z")
    #     image_new = Any2One(fbp(sinogram_new, geo_full))
    #     ssim,mse,psnr = ssim_mse_psnr(image_full, image_new)
    #     if ssim>s:
    #         print("SSIM:", weg)
    #         s = ssim
    #         weg_s = weg
    #     if psnr>p:
    #         print("PSNR:", weg)
    #         p = psnr
    #         weg_p = weg
    # print("SSIM:", weg_s)
    # print("Psnr:", weg_p)

if __name__ == "__main__":
    parsers = InitParser()
    main(parsers)
    print("Run Done")

