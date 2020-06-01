Change  
=====  
***Change 4***  
——————————  
  * `location`: TITAN-2:7   
  * `iter`: 4  
  * `version`: v1  
  * `batch_size`:"train": 8, "val": 8, "test": 1  
  * `k_size`：5  
  * `lr`:0.00001  
  * `step_size`:30  
  * `epoch_num`:300  
  `Train Time`:23.0689h -   epoch  


***Change 4***  
——————————  
  * `location`: TITAN-2:7   
  * `iter`: 3  
  * `version`: v1  
  * `batch_size`:"train": 8, "val": 8, "test": 1  
  * `k_size`：5  
  * `lr`:0.00001  
  * `step_size`:30  
  * `epoch_num`:300  
  `Train Time`:23.0689h -   epoch  


***Change 3***  
——————————  
  * `location`: TITAN-2:2   
  * `iter`: 2  
  * `version`: v1  
  * `batch_size`:"train": 8, "val": 8, "test": 1  
  * `k_size`：5  
  * `lr`:0.00001  
  * `step_size`:100  
  * `epoch_num`:300  
  `Train Time`:139.6609h - 290 epoch


***Change 2***  
——————————  
  * `location`: TITAN-2:2    
  * `iter`: 1  
  * `version`: v3  
  * `batch_size`:"train": 8, "val": 8, "test": 1  
  * `k_size`：5  
  * `lr`:0.00001  
  * `step_size`:100  
  * `epoch_num`:300  

***Change 1***  
——————————  
  * `location`: TITAN-2:3    
  * `iter`: 1  
  * `version`: v2  
  * `batch_size`:"train": 10, "val": 10, "test": 1  
  * `k_size`：3  
  * `lr`:0.00001  
  * `step_size`:100  
  * `epoch_num`:300  
  * `Train Time`:23.0689h
  * Change the out lay  


Base Program  
=====  
  ***Model Parameters***   
  * ResUnet  
  `input_channels`：1   
  `output_channels`：1  
  `k_size`：3  
  `bilinear`：True  
  
  ***Train Parameters***  
  `location`: TITAN-2:3  
  `iter`: 1  
  `version`: v1  
  `batch_size`:"train": 10, "val": 10, "test": 1    
  `epoch_num`:300   
  `sparse_view`:60   
  `full_view`:1160   
  
  ***Optimizer***  
  `lr`:0.00001   
  `momentum`:0.9   
  `weight_decay`:0.0    

  ***Scheduler***   
  `step_size`:30   
  `gamma`:0.5     


Change  
=====  
           `SSIM`          `MSE`          `PSNR`        `LOSS`  
 ***Iter 1 ***          
Sparse: 9.01664996e-01 4.23027501e-03 7.23836540e+01 4.23027482e-03  
Pred:   9.57053556e-01 4.03123719e-04 8.24934096e+01 4.03123617e-04  
 ***Iter 2 ***          
Sparse: 9.66634647e-01 4.36002759e-04 8.19384879e+01 4.36002767e-04  
Pred:   9.69682575e-01 3.49178027e-04 8.30219784e+01 3.49178008e-04  
 ***Iter 3 ***    
Sparse: 9.75553390e-01 3.68908455e-04 8.27288086e+01 3.68908572e-04
Pred:   9.74659132e-01 3.00867725e-04 8.38446652e+01 3.00867687e-04