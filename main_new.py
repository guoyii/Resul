import torch
import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt 
from PIL import Image
import matplotlib

from main_function import check_dir, build_geo
from datasets_new import BuildDataSet
# from torch.utils.data import DataLoader

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
    # index = np.random.randint(low=0, high=223)
    index = 0
    datasets = BuildDataSet(args.data_root_path, args.test_folder, geo_full, geo_sparse, None, "test")

    data_length = len(datasets)
    if not data_length == args.data_length:
        print("Args.data_length is wrong!")
        sys.exit(0)
    sample = datasets[index]
    
    image_true =sample["image_true"]
    image_full = sample["image_full"]
    image_sparse = sample["image_sparse"]
    image_inter = sample["image_inter"]
    sinogram_sparse = sample["sinogram_sparse"]
    image_temp = sample["image_temp"]
    # print(type(image_temp))
    # save_image = Image.fromarray(image_temp)
    # print(type(save_image))
    # save_image.save("V:/users/gy/MyProject/Resul/results/image/UseInNet.png")
    
    matplotlib.image.imsave("V:/users/gy/MyProject/Resul/results/image/UseInNet.png", image_temp, cmap="gray")

    plt.figure()
    plt.subplot(231), plt.xticks([]), plt.yticks([]), plt.imshow(image_true, cmap="gray"),       plt.title("image_true")
    plt.subplot(232), plt.xticks([]), plt.yticks([]), plt.imshow(image_full, cmap="gray"),       plt.title("image_full")
    plt.subplot(233), plt.xticks([]), plt.yticks([]), plt.imshow(image_sparse, cmap="gray"),     plt.title("image_sparse")
    plt.subplot(234), plt.xticks([]), plt.yticks([]), plt.imshow(image_inter, cmap="gray"),   plt.title("image_inter")
    plt.subplot(235), plt.xticks([]), plt.yticks([]), plt.imshow(sinogram_sparse, cmap="gray"),   plt.title("sinogram_sparse")
    plt.subplot(236), plt.xticks([]), plt.yticks([]), plt.imshow(image_temp, cmap="gray"),       plt.title("image_temp")
    plt.show()
   
     
if __name__ == "__main__":
    parsers = InitParser()
    main(parsers)
    print("Run Done")