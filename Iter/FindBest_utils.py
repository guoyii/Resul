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
from utils import rec_other
from model_basic1 import UnetDa
from torch.autograd import Variable
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
class ModelInit(object):
    def __init__(self):
        self.input_channels=1
        self.output_channels=1
        self.k_size=3
        self.bilinear=True

class Unetda(object):
    def __init__(self, root_path):
        self.result_path_1 = root_path + "/UnetDa/results/v2/model/UnetDa_E297_val_Best.pkl"
        self.args = ModelInit()
        self.model = UnetDa(self.args)
        self.model = model_updata(self.model, self.result_path_1)

    def __call__(self, image_sparse):
        start_time = time.time()
        image_pred = pred_sample(image_sparse, self.model)
        print("Unet Time:{:.4f}s".format(time.time()-start_time))
        return image_pred+image_sparse

## Build different views of geo
##***********************************************************************************************************
def build_geo(views, image_size=128):
    geo = {"nVoxelX": image_size, "nVoxelY": image_size, 
       "sVoxelX": image_size, "sVoxelY": image_size, 
       "dVoxelX": 1.0, "dVoxelY": 1.0, 
       "sino_views": views, 
       "nDetecU": 736, "sDetecU": 736.0,
       "dDetecU": 1.0, "DSD": 600.0, "DSO": 550.0, "DOD": 50.0,
       "offOriginX": 0.0, "offOriginY": 0.0, 
       "offDetecU": 0.0,
       "start_angle": 0, "end_angle": np.pi,
       "accuracy": 0.5, "mode": "parallel", 
       "extent": 3,
       "COR": 0.0}
    return geo


## Check the path
##***********************************************************************************************************
def check_dir(path):
	if not os.path.exists(path):
		try:
			os.mkdir(path)
		except:
			os.makedirs(path)


## show ssim mse psnr
##******************************************************************************************************************************
def ssim_mse_psnr(image_true, image_test):
    image_true = Any2One(image_true)
    image_test = Any2One(image_test)
    mse = compare_mse(image_true, image_test)
    ssim = compare_ssim(image_true, image_test)
    psnr = compare_psnr(image_true, image_test, data_range=255)
    return ssim, mse, psnr

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

def show_image(image):
    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap="gray")
    plt.show()

def Any2One(image):
    image_max = np.max(image)
    image_min = np.min(image)
    return (image-image_min)/(image_max-image_min)

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
    
    result_path = root_path + "/Resul/Iter/Avg_Result"
    check_dir(result_path)

    full_result = np.zeros((225,4))
    inter_result = np.zeros((225,4))
    sparse_result = np.zeros((225,4))
    iter1_result = np.zeros((225,4))
    iter2_result = np.zeros((225,4))
    iter3_result = np.zeros((225,4))
    iter4_result = np.zeros((225,4))
    iter5_result = np.zeros((225,4))
    iter6_result = np.zeros((225,4))
    iter7_result = np.zeros((225,4))
    iter8_result = np.zeros((225,4))
    unet_result = np.zeros((225,4))
    cgls_result = np.zeros((225,4))
    sart_result = np.zeros((225,4))

    

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

    result_path_4 = root_path + "/IterDa/results/Iter_4/v2/model/IterDa_E119_val_Best.pth"
    model_i4 = torch.load(result_path_4, map_location=torch.device('cpu'))

    result_path_5 = root_path + "/IterDa/results/Iter_4/v2/model/IterDa_E119_val_Best.pth"
    model_i5 = torch.load(result_path_5, map_location=torch.device('cpu'))

    result_path_6 = root_path + "/IterDa/results/Iter_4/v2/model/IterDa_E119_val_Best.pth"
    model_i6 = torch.load(result_path_6, map_location=torch.device('cpu'))

    result_path_7 = root_path + "/IterDa/results/Iter_4/v2/model/IterDa_E119_val_Best.pth"
    model_i7 = torch.load(result_path_7, map_location=torch.device('cpu'))

    result_path_8 = root_path + "/IterDa/results/Iter_4/v2/model/IterDa_E119_val_Best.pth"
    model_i8 = torch.load(result_path_8, map_location=torch.device('cpu'))

    unet = Unetda(root_path)

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
        
        image_true = image_full

        loss = criterion(torch.from_numpy(image_true), torch.from_numpy(image_full))
        ssim,mse,psnr = ssim_mse_psnr(image_true, image_full)
        full_result[i] = [loss,ssim,mse,psnr]
        del loss,ssim,mse,psnr
    
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

        image_i5 = iter_one(model_i5, sinogram_sparse, image_i4, geo_full)
        loss = criterion(torch.from_numpy(image_true), torch.from_numpy(image_i5))
        ssim,mse,psnr = ssim_mse_psnr(image_true, image_i5)
        iter5_result[i] = [loss,ssim,mse,psnr]
        del loss,ssim,mse,psnr 

        image_i6 = iter_one(model_i6, sinogram_sparse, image_i5, geo_full)
        loss = criterion(torch.from_numpy(image_true), torch.from_numpy(image_i6))
        ssim,mse,psnr = ssim_mse_psnr(image_true, image_i6)
        iter6_result[i] = [loss,ssim,mse,psnr]
        del loss,ssim,mse,psnr 

        image_i7 = iter_one(model_i7, sinogram_sparse, image_i6, geo_full)
        loss = criterion(torch.from_numpy(image_true), torch.from_numpy(image_i7))
        ssim,mse,psnr = ssim_mse_psnr(image_true, image_i7)
        iter7_result[i] = [loss,ssim,mse,psnr]
        del loss,ssim,mse,psnr 

        image_i8 = iter_one(model_i8, sinogram_sparse, image_i7, geo_full)
        loss = criterion(torch.from_numpy(image_true), torch.from_numpy(image_i8))
        ssim,mse,psnr = ssim_mse_psnr(image_true, image_i8)
        iter8_result[i] = [loss,ssim,mse,psnr]
        del loss,ssim,mse,psnr 

        image_unet = unet(image_sparse)
        loss = criterion(torch.from_numpy(image_true), torch.from_numpy(image_unet))
        ssim,mse,psnr = ssim_mse_psnr(image_true, image_unet)
        unet_result[i] = [loss,ssim,mse,psnr]
        del loss,ssim,mse,psnr 

        image_cgls =  rec_other(sinogram_sparse, geo_sparse, "CGLS", 30)
        loss = criterion(torch.from_numpy(image_true), torch.from_numpy(image_cgls))
        ssim,mse,psnr = ssim_mse_psnr(image_true, image_cgls)
        cgls_result[i] = [loss,ssim,mse,psnr]
        del loss,ssim,mse,psnr 

        image_sart =  rec_other(sinogram_sparse, geo_sparse, "SART", 100)
        loss = criterion(torch.from_numpy(image_true), torch.from_numpy(image_sart))
        ssim,mse,psnr = ssim_mse_psnr(image_true, image_sart)
        sart_result[i] = [loss,ssim,mse,psnr]
        del loss,ssim,mse,psnr 

            
    for i in range(4):
        full_result[224, i] = np.mean(full_result[0:224,i])
        sparse_result[224, i] = np.mean(sparse_result[0:224,i])
        inter_result[224, i] = np.mean(inter_result[0:224,i])
        iter1_result[224, i] = np.mean(iter1_result[0:224,i])
        iter2_result[224, i] = np.mean(iter2_result[0:224,i])
        iter3_result[224, i] = np.mean(iter3_result[0:224,i])
        iter4_result[224, i] = np.mean(iter4_result[0:224,i])
        iter5_result[224, i] = np.mean(iter5_result[0:224,i])
        iter6_result[224, i] = np.mean(iter6_result[0:224,i])
        iter7_result[224, i] = np.mean(iter7_result[0:224,i])
        iter8_result[224, i] = np.mean(iter8_result[0:224,i]) 
        unet_result[224, i] = np.mean(unet_result[0:224,i]) 
        cgls_result[224, i] = np.mean(cgls_result[0:224,i]) 
        sart_result[224, i] = np.mean(sart_result[0:224,i]) 

    """
    np.save("filename.npy",a)
    b = np.load("filename.npy")
    """
    result_path = result_path + "/Avg"
    check_dir(result_path)
    np.save(result_path + "/full_result.npy", full_result)
    np.save(result_path + "/sparse_result.npy", sparse_result)
    np.save(result_path + "/inter_result.npy", inter_result)
    np.save(result_path + "/iter1_result.npy", iter1_result)
    np.save(result_path + "/iter2_result.npy", iter2_result)
    np.save(result_path + "/iter3_result.npy", iter3_result)
    np.save(result_path + "/iter4_result.npy", iter4_result)
    np.save(result_path + "/iter5_result.npy", iter5_result)
    np.save(result_path + "/iter6_result.npy", iter6_result)
    np.save(result_path + "/iter7_result.npy", iter7_result)
    np.save(result_path + "/iter8_result.npy", iter8_result)
    np.save(result_path + "/unet_result.npy",  unet_result)
    np.save(result_path + "/cgls_resul.npy",   cgls_result)
    np.save(result_path + "/sart_result.npy",  sart_result)

    
    avg_result = np.zeros((14,4))
    avg_result[0] = full_result[224]
    avg_result[1] = sparse_result[224]
    avg_result[2] = inter_result[224]
    avg_result[3] = iter1_result[224]
    avg_result[4] = iter2_result[224]
    avg_result[5] = iter3_result[224] 
    avg_result[6] = iter4_result[224]
    avg_result[7] = iter5_result[224]
    avg_result[8] = iter6_result[224]
    avg_result[9] = iter7_result[224]
    avg_result[10] = iter8_result[224]
    avg_result[11] = unet_result[224]
    avg_result[12] = cgls_result[224]
    avg_result[13] = sart_result[224]


    np.save(result_path + "/avg_result.npy", avg_result)

    print("SSIM   MSE   PSNR   LOSS")
    print(avg_result)
    print("Test completed ! Time is {:.4f}min".format((time.time() - time_all_start)/60)) 

    """
    [0.00000000e+00 1.00000000e+00 0.00000000e+00            inf]
 [4.72984394e-03 7.91025610e-01 2.30095386e-02 6.62099719e+01]
 [6.12000076e-04 9.23467585e-01 1.97412800e-03 7.68956927e+01]
 [4.51036315e-04 9.63444561e-01 9.10058122e-04 7.98493704e+01]
 [3.95183817e-04 9.72685791e-01 8.18188347e-04 8.04539148e+01]
 [3.58240236e-04 9.73411112e-01 7.20953789e-04 8.08331998e+01]
 [3.46064048e-04 9.73323251e-01 6.77338956e-04 8.10326241e+01]
 [3.51778010e-04 9.71409437e-01 6.64486808e-04 8.09922027e+01]
 [3.69947681e-04 9.68357883e-01 6.77732932e-04 8.07784174e+01]
 [3.97791559e-04 9.64590035e-01 7.06877310e-04 8.04921733e+01]
 [4.33512665e-04 9.60324999e-01 7.52659033e-04 8.01456734e+01]
 [5.30288133e-04 8.99273850e-01 2.10201808e-03 7.58040014e+01]
 [2.09878638e-02 6.17631779e-01 6.55161012e-02 6.10476535e+01]
 [4.74765755e-02 6.32604640e-01 8.56624038e-02 5.98193813e+01]
    """

