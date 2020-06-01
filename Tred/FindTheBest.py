import numpy as np 
import time
import os 
import torch 
import sys 
import matplotlib.pyplot as plt
from datasets import BuildDataSet
from torch.utils.data import DataLoader
from utils import build_geo, check_dir, fbp, SinoInter, ssim_mse_psnr


def train(dataloaders, geo_full, option, zOf, ref, weg):
    ssim_all = 0
    mse_all = 0
    psnr_all = 0
    for i, batch in enumerate(dataloaders):
        if i<10:
            print("Sample {}/{}".format(i+1, 224))
            image_full = batch[ref][0].numpy()
            sinogram_inter = batch["sinogram_inter"][0].numpy()

            sinogram_new = SinoInter(sinogram_inter, geo_full, weg, option, zOf)
            image_new = fbp(sinogram_new, geo_full)
            ssim, mse, psnr = ssim_mse_psnr(image_full, image_new)
            ssim_all = ssim_all + ssim
            mse_all = mse_all + mse
            psnr_all = psnr_all + psnr
        else:
            pass
    ssim_avg = ssim_all/224
    mse_avg = mse_all/224
    psnr_avg = psnr_all/224
    return ssim_avg, mse_avg, psnr_avg


def test_f(dataloaders, geo_full, option, zOf, ref):
    weg_num = 500
    wegs = np.linspace(2, 7, weg_num)
    ssim_max = 0
    mse_min = 10000
    psnr_max = 0

    ssim_weg = -1
    mse_weg = -1
    psnr_weg = -1

    for i,weg in enumerate(wegs):
        print("Now weg={}, {}/{}".format(weg, i+1, weg_num))
        ssim,mse,psnr = train(dataloaders, geo_full, option, zOf, ref, weg)
        if ssim>ssim_max:
            ssim_max = ssim
            ssim_weg = weg
        if mse<mse_min:
            mse_min = mse
            mse_weg = weg
        if psnr>psnr_max:
            psnr_max = psnr
            psnr_weg = weg 
    result = np.array([[ssim_weg, ssim_max],[mse_weg, mse_min],[psnr_weg, psnr_max]])
    return result 


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
    else:
        print("Using CPU")

    geo_full = build_geo(args.full_view)
    geo_sparse = build_geo(args.sparse_view)

    datasets = BuildDataSet(args.data_root_path, args.test_folder, geo_full, geo_sparse, None, "test")
    kwargs = {"num_workers": args.num_workers, "pin_memory": True if args.mode is "train" else False}
    dataloaders = DataLoader(datasets, args.batch_size, shuffle=args.is_shuffle, **kwargs)
    
    refs = ("image_true", "image_full")
    options = ("sinogram_LineInter", "other")
    zOfs = ("other", "z")
     
    phase = "find"

    for ref in refs:
        for option in options:
            for zOf in zOfs:
                result_name = ref+ "_" + option + "_" + zOf
                result_path = args.root_path + "/Resul/Tred/FindTheBest_Result"

                if phase is "find":
                    print("Ref:{}, Option:{}, ZOf:{}".format(ref, option, zOf))
                    result = test_f(dataloaders, geo_full, option, zOf, ref)
                    print(result)
                    check_dir(result_path)
                    np.save(result_path + result_name + ".npy", result)
                elif phase is "read":
                    result =  np.load(result_path + result_name + ".npy")
                    print("Result Name:{}".format(result_name))
                    print(result)
                else:
                    print("There are some wrong")
                    sys.exit(0)
                
    
    print("Run Done")

if __name__ == "__main__":
    parsers = InitParser()
    main(parsers)