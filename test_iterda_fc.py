import os, sys
import time
import copy
import math
import pickle
import astra
import numpy as np
import scipy.io as sio
from scipy.io import loadmat 
import torch
from torch.autograd import Variable
from exhibit_function import ssim_mse_psnr
from main_function import check_dir
from model_basic import ResUnet
from init import ModelInit

## Load model
##******************************************************************************************************************************
def model_updata(model, model_path):
    print("\nOld model path：{}".format(model_path))
    if os.path.isfile(model_path):
        print("Loading previously trained network...")
        checkpoint = torch.load(model_path, map_location = lambda storage, loc: storage)
        model_dict = model.state_dict()
        checkpoint =  {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
        del checkpoint
        torch.cuda.empty_cache()
        print("Loading Done!\n")
        return model
    else:
        print("\nLoading Fail!\n")
        sys.exit(0)

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

def iter_one(model, sinogram_sparse, image_input, geo):
    image_pred = pred_sample(image_input, model)
    sinogram_pred = project(image_pred, geo)
    sinogram_new = updata_sinogram(sinogram_sparse, sinogram_pred)
    image_new = fbp(sinogram_new, geo)
    return image_new


def test_model(dataloaders, geo_full, geo_sparse):
    """
    ***********************************************************************************************************
    初始化
    ***********************************************************************************************************
    """ 
    criterion = torch.nn.MSELoss()
    if torch.cuda.is_available():
        root_path = "/mnt/tabgha/users/gy/MyProject"
    else:
        root_path = "V:/users/gy/MyProject"
    
    result_path = root_path + "/Resul/results/SMPL_test"
    check_dir(result_path)

    full_result = np.zeros((225,4))
    inter_result = np.zeros((225,4))
    sparse_result = np.zeros((225,4))
    iter1_result = np.zeros((225,4))
    iter2_result = np.zeros((225,4))
    iter3_result = np.zeros((225,4))
    iter4_result = np.zeros((225,4))
    

    """
    ***********************************************************************************************************
    读取模型
    ***********************************************************************************************************
    """ 
    result_path_1 = root_path + "/IterDa/results/Iter_1/v3/model/IterDa_E199_val_Best.pth"
    model_i1 = torch.load(result_path_1, map_location=torch.device('cpu'))

    result_path_2 = root_path + "/IterDa/results/Iter_2/v1/model/IterDa_E281_val_Best.pth"
    model_i2 = torch.load(result_path_2, map_location=torch.device('cpu'))

    result_path_3 = root_path + "/IterDa/results/Iter_3/v1/model/IterDa_E211_val_Best.pth"
    model_i3 = torch.load(result_path_3, map_location=torch.device('cpu'))

    result_path_4 = root_path + "/IterDa/results/Iter_4/v1/model/IterDa_E17_val_Best.pkl"
    modelparser = ModelInit()
    model_i4 = ResUnet(modelparser)
    model_i4 = model_updata(model_i4, result_path_4)

    ## SSIM MSE PSN LOSS
    print("**************  Test  ****************")
    time_all_start = time.time()
    for i, batch in enumerate(dataloaders):
        print("Now testing {} sample......".format(i))

        image_true =batch["image_true"][0].numpy()
        image_full = batch["image_full"][0].numpy()
        image_sparse = batch["image_sparse"][0].numpy()
        image_inter = batch["image_inter"][0].numpy()
        sinogram_sparse = batch["sinogram_sparse"][0].numpy()
        
        loss = criterion(torch.from_numpy(image_true), torch.from_numpy(image_full))
        ssim,mse,psnr = ssim_mse_psnr(image_true, image_full)
        full_result[i] = [loss,ssim,mse,psnr]
        del loss,ssim,mse,psnr

        image_true = image_full

        loss = criterion(torch.from_numpy(image_true), torch.from_numpy(image_sparse))
        ssim,mse,psnr = ssim_mse_psnr(image_true, image_sparse)
        sparse_result[i] = [loss,ssim,mse,psnr]
        del loss,ssim,mse,psnr

        loss = criterion(torch.from_numpy(image_true), torch.from_numpy(image_inter))
        ssim,mse,psnr = ssim_mse_psnr(image_true, image_inter)
        inter_result[i] = [loss,ssim,mse,psnr]
        del loss,ssim,mse,psnr
        
        image_i1 = iter_one(model_i1, sinogram_sparse, image_sparse, geo_full)
        loss = criterion(torch.from_numpy(image_true), torch.from_numpy(image_i1))
        ssim,mse,psnr = ssim_mse_psnr(image_true, image_i1)
        iter1_result[i] = [loss,ssim,mse,psnr]
        del loss,ssim,mse,psnr

        image_i2 = iter_one(model_i2, sinogram_sparse, image_i1, geo_full)
        loss = criterion(torch.from_numpy(image_true), torch.from_numpy(image_i2))
        ssim,mse,psnr = ssim_mse_psnr(image_true, image_i2)
        iter2_result[i] = [loss,ssim,mse,psnr]
        del loss,ssim,mse,psnr

        image_i3 = iter_one(model_i3, sinogram_sparse, image_i2, geo_full)
        loss = criterion(torch.from_numpy(image_true), torch.from_numpy(image_i3))
        ssim,mse,psnr = ssim_mse_psnr(image_true, image_i3)
        iter3_result[i] = [loss,ssim,mse,psnr]
        del loss,ssim,mse,psnr 

        image_i4 = iter_one(model_i4, sinogram_sparse, image_i3, geo_full)
        loss = criterion(torch.from_numpy(image_true), torch.from_numpy(image_i4))
        ssim,mse,psnr = ssim_mse_psnr(image_true, image_i4)
        iter4_result[i] = [loss,ssim,mse,psnr]
        del loss,ssim,mse,psnr 

            
    for i in range(4):
        full_result[224, i] = np.mean(full_result[0:224,i])
        sparse_result[224, i] = np.mean(sparse_result[0:224,i])
        inter_result[224, i] = np.mean(inter_result[0:224,i])
        iter1_result[224, i] = np.mean(iter1_result[0:224,i])
        iter2_result[224, i] = np.mean(iter2_result[0:224,i])
        iter3_result[224, i] = np.mean(iter3_result[0:224,i])
        iter4_result[224, i] = np.mean(iter4_result[0:224,i])

    """
    np.save("filename.npy",a)
    b = np.load("filename.npy")
    """
    result_path = result_path + "/ref_full"
    check_dir(result_path)
    np.save(result_path + "/full_result.npy", full_result)
    np.save(result_path + "/sparse_result.npy", sparse_result)
    np.save(result_path + "/inter_result.npy", inter_result)
    np.save(result_path + "/iter1_result.npy", iter1_result)
    np.save(result_path + "/iter2_result.npy", iter2_result)
    np.save(result_path + "/iter3_result.npy", iter3_result)
    np.save(result_path + "/iter4_result.npy", iter4_result)
    
    avg_result = np.zeros((10,4))
    avg_result[0] = full_result[224]
    avg_result[1] = sparse_result[224]
    avg_result[2] = inter_result[224]
    avg_result[3] = iter1_result[224]
    avg_result[4] = iter2_result[224]
    avg_result[5] = iter3_result[224]
    avg_result[6] = iter4_result[224]
    np.save(result_path + "/avg_result.npy", avg_result)

    print("SSIM   MSE   PSNR   LOSS")
    print(avg_result)
    print("Test completed ! Time is {:.4f}min".format((time.time() - time_all_start)/60)) 

    """
    结果：
    
    [1.77097722e-04 9.92070549e-01 1.77097722e-04 8.62013946e+01]
    [4.17583170e-03 8.77348294e-01 4.17583171e-03 7.28738276e+01]
    [1.05955826e-03 9.58950696e-01 1.05955826e-03 7.85226514e+01]
    [8.47404992e-04 9.41867088e-01 8.47404993e-04 7.95971532e+01]
    [8.36379705e-04 9.46036436e-01 8.36379704e-04 7.97085470e+01]
    [8.03258379e-04 9.47388637e-01 8.03258380e-04 7.98992522e+01]
    [8.05562885e-04 9.42337164e-01 8.05562886e-04 7.98213296e+01]

    [1.60660733e-04 9.93630278e-01 1.60660733e-04 8.63808272e+01]
    [4.65445786e-03 8.83138937e-01 4.65445786e-03 7.21004770e+01]
    [6.42473313e-04 9.72908102e-01 6.42473314e-04 8.05875604e+01]
    [3.47941913e-04 9.66559699e-01 3.47941913e-04 8.29944000e+01]
    [3.09003061e-04 9.74368979e-01 3.09003061e-04 8.35755808e+01]
    [2.86913073e-04 9.74990774e-01 2.86913073e-04 8.38923037e+01]
    [2.94612524e-04 9.67710351e-01 2.94612524e-04 8.36807699e+01]

    """
