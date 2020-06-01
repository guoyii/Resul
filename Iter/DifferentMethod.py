import numpy as np 
import time
import os 
import torch 
import sys 
import matplotlib
import matplotlib.pyplot as plt
from datasets import BuildDataSet
from torch.utils.data import DataLoader
from utils import build_geo, check_dir, fbp, ssim_mse_psnr, rec_other
from FindBest_utils import iter_one, pred_sample
from utils import Any2One

class Iterda(object):
    def __init__(self, root_path, sinogram_sparse, geo_full):
        self.result_path_1 = root_path + "/IterDa/results/Iter_1/v3/model/IterDa_E199_val_Best.pth"
        self.model_i1 = torch.load(self.result_path_1, map_location=torch.device('cpu'))

        self.result_path_2 = root_path + "/IterDa/results/Iter_2/v1/model/IterDa_E281_val_Best.pth"
        self.model_i2 = torch.load(self.result_path_2, map_location=torch.device('cpu'))

        self.result_path_3 = root_path + "/IterDa/results/Iter_3/v1/model/IterDa_E211_val_Best.pth"
        self.model_i3 = torch.load(self.result_path_3, map_location=torch.device('cpu'))

        self.sinogram_sparse = sinogram_sparse
        self.geo_full = geo_full
    def __call__(self, image_sparse):
        start_time = time.time()
        image_i1 = iter_one(self.model_i1, self.sinogram_sparse, image_sparse, self.geo_full)
        image_i2 = iter_one(self.model_i2, self.sinogram_sparse, image_i1, self.geo_full)
        image_i3 = iter_one(self.model_i3, self.sinogram_sparse, image_i2, self.geo_full)
        print("Iter Time:{:.4f}s".format(time.time()-start_time))
        return image_i3

## sparse transform
##***********************************************************************************************************
def sparse_view_f(sino_true,  view_origin=1160, view_sparse=60):
   view_index = (np.linspace(0, view_origin-1, view_sparse)).astype(np.int32)
   return sino_true[view_index, :]

class InitParser(object):
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        self.mode = "test"
        self.num_workers = 20
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
            
        self.test_folder = {"patients": "L067", "SliceThickness": "full_3mm"}

def main(args):
    if args.use_cuda:
        print("Using GPU")
        root_path = "/mnt/tabgha/users/gy/MyProject"
    else:
        print("Using CPU")
        root_path = "V:/users/gy/MyProject"

    geo_full = build_geo(args.full_view)
    geo_sparse = build_geo(args.sparse_view)

    datasets = BuildDataSet(args.data_root_path, args.test_folder, geo_full, geo_sparse, None, "test")
    sample = datasets[0]  
    image_true = sample["image_true"] 
    image_full = sample["image_full"] 
    image_sparse = sample["image_sparse"] 
    sinogram_sparse = sample["sinogram_sparse"]

    image_pred =  rec_other(sinogram_sparse, geo_sparse, "CGLS", 30)
    image_pred =  rec_other(sinogram_sparse, geo_sparse, "SART", 100)
    image_pred = Any2One(image_pred)


    


    plt.figure()
    plt.subplot(231), plt.xticks([]), plt.yticks([]), plt.imshow(image_true, cmap="gray"),       plt.title("image_true")
    plt.subplot(232), plt.xticks([]), plt.yticks([]), plt.imshow(image_full, cmap="gray"),       plt.title("image_full")
    plt.subplot(233), plt.xticks([]), plt.yticks([]), plt.imshow(image_sparse, cmap="gray"),     plt.title("image_sparse")
    plt.subplot(234), plt.xticks([]), plt.yticks([]), plt.imshow(image_pred, cmap="gray"),   plt.title("image_inter")
    # plt.subplot(235), plt.xticks([]), plt.yticks([]), plt.imshow(sinogram_sparse, cmap="gray"),   plt.title("sinogram_sparse")
    # plt.subplot(236), plt.xticks([]), plt.yticks([]), plt.imshow(image_origin, cmap="gray"),       plt.title("image_origin")
    # plt.savefig("test.png")
    plt.show()
    
    print("Run Done")

# if __name__ == "__main__":
#     parsers = InitParser()
#     main(parsers)