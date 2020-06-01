import numpy as np 
import time
import os 
import torch 
import sys 
import astra
import copy
import matplotlib
import matplotlib.pyplot as plt
from datasets import BuildDataSet
from torch.utils.data import DataLoader
from utils import build_geo, check_dir, fbp, ssim_mse_psnr
from FindBest_utils import Any2One

## FBP
##***********************************************************************************************************
def fbp(sinogram, geo):
    vol_geom = astra.create_vol_geom(geo["nVoxelY"], geo["nVoxelX"], 
                                            -1*geo["sVoxelY"]/2, geo["sVoxelY"]/2, -1*geo["sVoxelX"]/2, geo["sVoxelX"]/2)
    proj_geom = astra.create_proj_geom(geo["mode"], geo["dDetecU"], geo["nDetecU"], 
                                                np.linspace(geo["start_angle"], geo["end_angle"], geo["sino_views"],False), geo["DSO"], geo["DOD"])
    if geo["mode"] is "parallel":
        proj_id = astra.create_projector("linear", proj_geom, vol_geom)
    elif geo["mode"] is "fanflat":
        proj_id = astra.create_projector("line_fanflat", proj_geom_full, vol_geom)

    
    rec_id = astra.data2d.create('-vol', vol_geom)
    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)

    cfg = astra.astra_dict('FBP')
    cfg['ProjectorId'] = proj_id
    cfg["FilterType"] = "Ram-Lak" 
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    
    
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    
    image_recon = astra.data2d.get(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    
    return image_recon

            
def project(image, geo):
    vol_geom = astra.create_vol_geom(geo["nVoxelY"], geo["nVoxelX"], 
                                            -1*geo["sVoxelY"]/2, geo["sVoxelY"]/2, -1*geo["sVoxelX"]/2, geo["sVoxelX"]/2)
    proj_geom = astra.create_proj_geom(geo["mode"], geo["dDetecU"], geo["nDetecU"], 
                                            np.linspace(geo["start_angle"], geo["end_angle"], geo["sino_views"],False), geo["DSO"], geo["DOD"])
    if geo["mode"] is "parallel":
        proj_id = astra.create_projector("linear", proj_geom, vol_geom)
    elif geo["mode"] is "fanflat":
        proj_id = astra.create_projector("line_fanflat", proj_geom, vol_geom)
    sinogram_id, sino = astra.create_sino(image, proj_id) 
    astra.data2d.delete(sinogram_id)
    sinogram = copy.deepcopy(sino)
    return sinogram
## sparse transform
##***********************************************************************************************************
def sparse_view_f(sino_true,  view_origin=1160, view_sparse=60):
   view_index = (np.linspace(0, view_origin-1, view_sparse)).astype(np.int32)
   return sino_true[view_index, :]

def updata_sinogram(sinogram_sparse, sinogram_pred):
    view_index = (np.linspace(0, sinogram_pred.shape[0]-1, sinogram_sparse.shape[0])).astype(np.int32)
    for i,index in enumerate(view_index):
        sinogram_pred[index] = sinogram_sparse[i]
    return sinogram_pred

def pred_sample(image_sparse, model):
    model.eval()
    with torch.no_grad():
        image_pred = model(torch.from_numpy(image_sparse).unsqueeze_(0).unsqueeze_(0)).numpy()
    return image_pred[0,0,:,:]

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

    result_path_1 = args.root_path + "/IterDa/results/Iter_1/v3/model/IterDa_E199_val_Best.pth"
    model_i1 = torch.load(result_path_1, map_location=torch.device('cpu'))

    datasets = BuildDataSet(args.data_root_path, args.test_folder, geo_full, geo_sparse, None, "test")
    sample = datasets[0]  
    image_true = sample["image_true"] 
    image_full = sample["image_full"] 
    image_sparse = sample["image_sparse"] 
    sinogram_sparse = sample["sinogram_sparse"]
    sinogram_full = sample["sinogram_full"]

    image_pred = pred_sample(image_sparse, model_i1)
    sinogram_full_pred = project(image_pred, geo_full)
    sinogram_sparse_pred = sparse_view_f(sinogram_full_pred, geo_full["sino_views"], geo_sparse["sino_views"])

    sinogram_updata = updata_sinogram(sinogram_sparse, sinogram_full_pred)
    image_updata = fbp(sinogram_updata, geo_full)

    # plt.figure()
    # plt.subplot(131), plt.xticks([]), plt.yticks([]), plt.imshow(sinogram_sparse, cmap="gray"),       plt.title("(a)", y=-2)
    # plt.subplot(132), plt.xticks([]), plt.yticks([]), plt.imshow(sinogram_sparse_pred, cmap="gray"),  plt.title("(b)", y=-2)
    # plt.subplot(133), plt.xticks([]), plt.yticks([]), plt.imshow(sinogram_sparse - sinogram_sparse_pred, cmap="gray"), plt.title("(c)", y=-2)
    # plt.savefig("V:/users/gy/MyProject/Resul/Tred/Image/image4-3.png")
    # plt.show()
    plt.figure()
    plt.subplot(131), plt.xticks([]), plt.yticks([]), plt.imshow(image_full, cmap="gray"),       plt.title("(a)", y=-2)
    plt.subplot(132), plt.xticks([]), plt.yticks([]), plt.imshow(image_pred, cmap="gray"),  plt.title("(b)", y=-2)
    plt.subplot(133), plt.xticks([]), plt.yticks([]), plt.imshow(image_updata, cmap="gray"), plt.title("(c)", y=-2)
    # plt.savefig("V:/users/gy/MyProject/Resul/Tred/Image/image4-3.png")
    plt.show()
    ssim,se,psnr = ssim_mse_psnr(Any2One(image_full), Any2One(image_pred))
    print("Pred:", ssim,se,psnr)
    ssim,se,psnr = ssim_mse_psnr(Any2One(image_full), Any2One(image_updata))
    print("Updata:", ssim,se,psnr)
    print("Run Done")

if __name__ == "__main__":
    parsers = InitParser()
    main(parsers)