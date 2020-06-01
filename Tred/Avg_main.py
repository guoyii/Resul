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
        
        self.batch_size= 50
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

    datasets = BuildDataSet(args.data_root_path, args.test_folder, geo_full, geo_sparse, None, "test")

    data_length = len(datasets)
    if not data_length == args.data_length:
        print("Args.data_length is wrong!")
        sys.exit(0)
        
    # s_sparse = 0
    # m_sparse = 0
    # p_sparse = 0
    s_inter = 0
    m_inter = 0
    p_inter = 0
    s_new = 0
    m_new = 0
    p_new = 0
    for i in range(224):
        print("Testing sample {}/224".format(i))
        sample = datasets[i]
        image_true = Any2One(sample["image_true"])
        image_full = Any2One(sample["image_full"])
        image_sparse = Any2One(sample["image_sparse"])
        image_inter = Any2One(sample["image_inter"])
        sinogram_full = sample["sinogram_full"]
        sinogram_sparse = sample["sinogram_sparse"]
        sinogram_inter = sample["sinogram_inter"]

        # sinogram_new = SinoInter(sinogram_inter, geo_full, -2, "sinogram_LineInter", "z")
        sinogram_new = newinter(sinogram_sparse)
        image_new = Any2One(fbp(sinogram_new, geo_full))

        # ssim, mse, psnr = ssim_mse_psnr(image_full, image_sparse)
        # s_sparse = s_sparse + ssim
        # m_sparse = m_sparse + mse
        # p_sparse = p_sparse + psnr
        ssim, mse, psnr = ssim_mse_psnr(image_full, image_inter)
        s_inter = s_inter + ssim
        m_inter = m_inter + mse
        p_inter = p_inter + psnr
        ssim, mse, psnr = ssim_mse_psnr(image_full, image_new)
        s_new = s_new + ssim
        m_new = m_new + mse
        p_new = p_new + psnr
    
    # print("Sparse:", s_sparse/224, m_sparse/224, p_sparse/224)
    print("Inter:", s_inter/224, m_inter/224, p_inter/224)
    print("New:", s_new/224, m_new/224, p_new/224)


if __name__ == "__main__": 
    parsers = InitParser()
    main(parsers)
    print("Run Done")

"""
-3
Sparse: 0.546166863573 0.0381710962769 14.3768192408
Inter: 0.752720364342 0.00701280885007 21.8058473806
New: 0.799321500503 0.00562279550962 23.0042998582

-4
Sparse: 0.546166863573 0.0381710962769 14.3768192408
Inter: 0.752720364342 0.00701280885007 21.8058473806
New: 0.786676864049 0.00686054342058 22.3499950581

-2
Sparse: 0.546166863573 0.0381710962769 14.3768192408
Inter: 0.752720364342 0.00701280885007 21.8058473806
New: 0.793474211917 0.00574750898174 22.8684491461

三次样条插值

牛顿插值
Inter: 0.719555794687 0.0094121240005 20.4703928275
New: 0.37321923796 0.109204813118 9.63104347964
"""