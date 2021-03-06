import numpy as np 
import astra
import os 
import numpy as np 
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import copy
import sys 


def Any2One(image):
    image_max = np.max(image)
    image_min = np.min(image)
    return (image-image_min)/(image_max-image_min)
    
## Inter Function
def SinoInter(sinogram_LineInter, geo_full, weg, option, zOf):
    """
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
    """
    angles = geo_full["end_angle"] - geo_full["start_angle"]
    angle = angles/geo_full["sino_views"]
    deta_length = geo_full["DSD"] * np.sin(angle)

    sinogram_inter_z = copy.copy(sinogram_LineInter)
    sinogram_inter_f = copy.copy(sinogram_LineInter)
    for i in range(geo_full["sino_views"]):
        if i==0:
            pass
        else:
            for index in range(geo_full["nDetecU"]):
                y = geo_full["nDetecU"]/2 - index
                temp = y/np.tan(angle)
                left_length = geo_full["DOD"] + temp
                start_index = geo_full["nDetecU"]/2 - left_length * np.sin(angle)
                end_index = start_index + deta_length
                end_index = int(end_index)
                start_index = int(start_index)
                if start_index < 0:
                    start_index = 0
                if end_index > geo_full["nDetecU"]-1:
                    end_index = geo_full["nDetecU"]-1 
                if option is "sinogram_LineInter":
                    avg = sinogram_LineInter[i-1, start_index:end_index+1].sum()
                else:
                    avg = sinogram_inter_z[i-1, start_index:end_index+1].sum()
                fenmu = end_index - start_index+2
                sinogram_inter_z[i, index] = (avg*(fenmu-weg)/(fenmu-1) + weg*sinogram_inter_z[i, index])/fenmu
    if zOf is "z":
        return sinogram_inter_z
    else:
        for i in range(geo_full["sino_views"]):
            i = geo_full["sino_views"]-1-i
            if i==geo_full["sino_views"]-1:
                pass
            else:
                for index in range(geo_full["nDetecU"]):
                    y = geo_full["nDetecU"]/2 - index
                    temp = geo_full["DOD"] * np.tan(angle/2)
                    end_index = geo_full["nDetecU"]/2 - ((y-temp)*np.cos(angle)-temp)
                    start_index = int(end_index - deta_length)
                    end_index = int(end_index)
                    if start_index < 0:
                        start_index = 0
                    if end_index > geo_full["nDetecU"]-1:
                        end_index = geo_full["nDetecU"]-1 
                    if option is "sinogram_LineInter":
                        avg = sinogram_LineInter[i+1, start_index:end_index+1].sum()
                    else:
                        avg = sinogram_inter_f[i+1, start_index:end_index+1].sum()
                    fenmu = end_index - start_index+2
                    sinogram_inter_f[i, index] = (avg*(fenmu-weg)/(fenmu-1) + weg*sinogram_inter_f[i, index])/fenmu
        return (sinogram_inter_z + sinogram_inter_f)/2


## show ssim mse psnr
##******************************************************************************************************************************
def ssim_mse_psnr(image_true, image_test):
    mse = compare_mse(image_true, image_test)
    ssim = compare_ssim(image_true, image_test)
    psnr = compare_psnr(image_true, image_test)
    return ssim, mse, psnr


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


## Build different views of geo
##***********************************************************************************************************
def build_geo(views, image_size=512):
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
