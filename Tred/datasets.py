import torch
import astra
import copy
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from datasets_function import RandomCrop, Normalize, Any2One, ToTensor
from datasets_function import findFiles, image_read
from datasets_function import my_extension, my_map_coordinates, sparse_view_f

## Basic datasets
##***********************************************************************************************************
class BasicData(Dataset):
    def __init__(self, data_root_path, folder, crop_size=None, trf_op=None, Dataset_name="test"):
        self.Dataset_name = Dataset_name
        self.trf_op = trf_op
        self.crop_size = crop_size
        self.fix_list = [Normalize(), Any2One(), ToTensor()]

        if Dataset_name is "train":
            self.image_paths = [findFiles(data_root_path + "/{}/{}/*.IMA".format(x, y)) for x in folder["patients"] for y in folder["SliceThickness"]]
            self.image_paths = [x for j in self.image_paths for x in j]
        else:
            self.image_paths = findFiles("{}/{}/{}/*.IMA".format(data_root_path, folder["patients"], folder["SliceThickness"]))
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = image_read(image_path)
        imgdata = image.pixel_array * image.RescaleSlope + image.RescaleIntercept

        if self.Dataset_name is not "train":     
            imgname = os.path.splitext(os.path.split(image_path)[1])[0]

        transform = transforms.Compose(self.fix_list)
        imgdata = transform(imgdata).numpy()  

        if self.Dataset_name is "train":
            return imgdata
        else: 
            return imgdata, imgname


class BuildDataSet(Dataset):
    def __init__(self, data_root_path, folder, geo_full, geo_sparse, pre_trans_img=None, Dataset_name="test"):
        self.Dataset_name = Dataset_name
        self.geo_full = geo_full
        self.geo_sparse = geo_sparse
        self.imgset = BasicData(data_root_path, folder, self.geo_full["nVoxelX"], pre_trans_img, Dataset_name)

        ## Full-----------------------------------------
        self.vol_geom_full = astra.create_vol_geom(self.geo_full["nVoxelY"], self.geo_full["nVoxelX"], 
                                            -1*self.geo_full["sVoxelY"]/2, self.geo_full["sVoxelY"]/2, -1*self.geo_full["sVoxelX"]/2, self.geo_full["sVoxelX"]/2)
        self.proj_geom_full = astra.create_proj_geom(self.geo_full["mode"], self.geo_full["dDetecU"], self.geo_full["nDetecU"], 
                                                np.linspace(self.geo_full["start_angle"], self.geo_full["end_angle"], self.geo_full["sino_views"],False), self.geo_full["DSO"], self.geo_full["DOD"])
        if self.geo_full["mode"] is "parallel":
            self.proj_id_full = astra.create_projector("linear", self.proj_geom_full, self.vol_geom_full)
        elif self.geo_full["mode"] is "fanflat":
            self.proj_id_full = astra.create_projector("line_fanflat", self.proj_geom_full, self.vol_geom_full)

        ## Sparse-----------------------------------------
        self.vol_geom_sparse = astra.create_vol_geom(self.geo_sparse["nVoxelY"], self.geo_sparse["nVoxelX"], 
                                            -1*self.geo_sparse["sVoxelY"]/2, self.geo_sparse["sVoxelY"]/2, -1*self.geo_sparse["sVoxelX"]/2, self.geo_sparse["sVoxelX"]/2)
        self.proj_geom_sparse = astra.create_proj_geom(self.geo_sparse["mode"], self.geo_sparse["dDetecU"], self.geo_sparse["nDetecU"], 
                                                np.linspace(self.geo_sparse["start_angle"], self.geo_sparse["end_angle"], self.geo_sparse["sino_views"],False), self.geo_sparse["DSO"], self.geo_sparse["DOD"])
        if self.geo_sparse["mode"] is "parallel":
            self.proj_id_sparse = astra.create_projector("linear", self.proj_geom_sparse, self.vol_geom_sparse)
        elif self.geo_sparse["mode"] is "fanflat":
            self.proj_id_sparse = astra.create_projector("line_fanflat", self.proj_geom_sparse, self.vol_geom_sparse)
        

    @classmethod
    def project(cls, image, proj_id):
        sinogram_id, sino = astra.create_sino(image, proj_id) 
        astra.data2d.delete(sinogram_id)
        sinogram = copy.deepcopy(sino)
        return sinogram

    @classmethod
    def fbp(cls, sinogram, proj_id, proj_geom, vol_geom):
        cfg = astra.astra_dict("FBP")
        cfg["ProjectorId"] = proj_id                                                  # possible values for FilterType:
        cfg["FilterType"] = "Ram-Lak"                                                 # none, ram-lak, shepp-logan, cosine, hamming, hann, tukey, lanczos,
                                                                                      # triangular, gaussian, barlett-hann, blackman, nuttall, blackman-harris,
                                                                                      # blackman-nuttall, flat-top, kaiser, parzen
        
        sinogram_id = astra.data2d.create("-sino", proj_geom, sinogram)               # astra.data2d.store(sinogram_id, sinogram)
        rec_id = astra.data2d.create("-vol", vol_geom)

        cfg["ReconstructionDataId"] = rec_id
        cfg["ProjectionDataId"] = sinogram_id
                                                                                      # Create and run the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        image_recon = astra.data2d.get(rec_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sinogram_id)
        astra.algorithm.delete(alg_id)
        return image_recon

    
    def __len__(self):
        return len(self.imgset)
    
    def __getitem__(self, idx):
        if self.Dataset_name is "train":
            image= self.imgset[idx]
        else:
            image, image_name = self.imgset[idx]
        
        # print("Image max {} min {} mean {}".format(np.array(image).max(), np.array(image).min(), np.array(image).mean()))
        sinogram_full = self.project(image, self.proj_id_full)
        sinogram_sparse = sparse_view_f(sinogram_full, self.geo_full["sino_views"], self.geo_sparse["sino_views"])
        sinogram_inter = my_map_coordinates(sinogram_sparse, (self.geo_full["sino_views"], self.geo_full["nDetecU"]), order=3)
        
        image_full = self.fbp(sinogram_full, self.proj_id_full, self.proj_geom_full, self.vol_geom_full)
        image_sparse = self.fbp(sinogram_sparse, self.proj_id_sparse, self.proj_geom_sparse, self.vol_geom_sparse)
        image_inter = self.fbp(sinogram_inter, self.proj_id_full, self.proj_geom_full, self.vol_geom_full)

        sample = {"image_true": image,
                "image_full": image_full,
                "image_sparse": image_sparse,
                "image_inter":image_inter,
                "sinogram_full":sinogram_full,
                "sinogram_sparse":sinogram_sparse,
                "sinogram_inter":sinogram_inter}
        return sample
