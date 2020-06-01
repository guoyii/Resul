import torch
import os
import sys
import numpy as np
import time

from main_function import check_dir, build_geo
from datasets import BuildDataSet
from torch.utils.data import DataLoader
from test_iterda_fc import test_model

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

    datasets = BuildDataSet(args.data_root_path, args.test_folder, geo_full, geo_sparse, None, "test")

    data_length = len(datasets)
    if not data_length == args.data_length:
        print("Args.data_length is wrong!")
        sys.exit(0)
    
    kwargs = {"num_workers": args.num_workers, "pin_memory": True if args.mode is "train" else False}
    dataloaders = DataLoader(datasets, args.batch_size, shuffle=args.is_shuffle, **kwargs)
    test_model(dataloaders, geo_full, geo_sparse)
    print("Run test_function.py Success!")
   
     
if __name__ == "__main__":
    parsers = InitParser()
    main(parsers)
    print("Run Done")