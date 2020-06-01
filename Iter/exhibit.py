import torch
import os
import sys
import numpy as np
import time

from FindBest_utils import build_geo, check_dir, test_model
from datasets import BuildDataSet
from torch.utils.data import DataLoader
from FindBest_utils import iter_one
import matplotlib.pyplot as plt 
from utils import rec_other, Any2One
from FindBest_utils import Unetda
from DifferentMethod import Iterda
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr



class InitParser(object):
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        self.num_workers = 50

        ## set parameters
        self.mode = "test"
        self.sparse_view = 60
        self.full_view = 1160
        self.is_shuffle = True if self.mode is "train" else False
        
        self.batch_size= 1
        self.data_length = 224

        # path setting
        if self.use_cuda:
            self.data_root_path = "/mnt/tabgha/users/gy/data/Mayo"
            self.root_path = "/mnt/tabgha/users/gy/MyProject"
        else:
            self.data_root_path = "V:/users/gy/data/Mayo"
            self.root_path = "V:/users/gy/MyProject"
        self.train_folder = {"patients": ["L096","L109","L143","L192","L286","L291","L310","L333", "L506"], "SliceThickness": ["full_3mm"]}
        self.test_folder = {"patients": "L067", "SliceThickness": "full_3mm"}
        self.val_folder = {"patients": "L067", "SliceThickness": "full_3mm"}


def main(args):
    if args.use_cuda:
        print("Using GPU")
    else: 
        print("Using CPU")

    if torch.cuda.is_available():
        root_path = "/mnt/tabgha/users/gy/MyProject"
    else:
        root_path = "V:/users/gy/MyProject"
    
    geo_full = build_geo(args.full_view)
    geo_sparse = build_geo(args.sparse_view)

    datasets = BuildDataSet(args.data_root_path, args.test_folder, geo_full, geo_sparse, None, "test")

    data_length = len(datasets)
    if not data_length == args.data_length:
        print("Args.data_length is wrong!")
        sys.exit(0)
    
    sample = datasets[0]
    image_true = sample["image_true"] 
    image_full = sample["image_full"] 
    image_sparse = sample["image_sparse"] 
    image_inter = sample["image_inter"]
    sinogram_sparse = sample["sinogram_sparse"]

    unetda = Unetda(root_path)
    iterda = Iterda(root_path, sinogram_sparse, geo_full)
    image_unet = unetda(image_sparse)
    image_iter = iterda(image_sparse)
    start_time = time.time() 
    image_cgls =  rec_other(sinogram_sparse, geo_sparse, "CGLS", 30)
    print("CGLS Time:{:.4f}s".format(time.time()-start_time))
    start_time = time.time() 
    image_sart =  rec_other(sinogram_sparse, geo_sparse, "SART", 100)
    print("SART Time:{:.4f}s".format(time.time()-start_time))

    image_1 = np.hstack((Any2One(image_full),
                        Any2One(image_sparse),
                        Any2One(image_cgls),
                        Any2One(image_sart),
                        Any2One(image_unet),
                        Any2One(image_iter),
                        ))
    image_2 = np.hstack((np.zeros(image_full.shape),
                        Any2One(image_full-image_sparse),
                        Any2One(image_full-image_cgls),
                        Any2One(image_full-image_sart),
                        Any2One(image_full-image_unet),
                        Any2One(image_full-image_iter),
                        ))
    image_all = np.vstack((image_1, image_2))

    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image_all, cmap="gray")
    plt.savefig("V:/users/gy/毕业设计/生物医学工程/论文/图表/第四章/图4-3.png")
    plt.show()
    ssim, mse, psnr = ssim_mse_psnr(image_full, image_sparse)
    print("Sparse:",ssim, mse, psnr)
    ssim, mse, psnr = ssim_mse_psnr(image_full, image_cgls)
    print("CGLS:",ssim, mse, psnr)
    ssim, mse, psnr = ssim_mse_psnr(image_full, image_sart)
    print("SART:",ssim, mse, psnr)
    ssim, mse, psnr = ssim_mse_psnr(image_full, image_unet)
    print("Unet:",ssim, mse, psnr)
    ssim, mse, psnr = ssim_mse_psnr(image_full, image_iter)
    print("Iter:",ssim, mse, psnr)

"""
Sparse: 0.767735589616 0.00940096144923 68.399080891
CGLS: 0.78799815288 0.00852202783728 68.8253741237
SART: 0.668496729185 0.0352627314792 62.6576441081
Unet: 0.918168804686 0.00174085477582 75.723178176
Iter: 0.969365970582 0.00182857178205 75.5096834728
"""


   
     
if __name__ == "__main__":
    parsers = InitParser()
    main(parsers)
    print("Run Done")