import matplotlib.pyplot as plt 
import numpy as np 
import os 
import torch 
from scipy.io import loadmat 

## show or return loss
##******************************************************************************************************************************
def show_loss(loss_name, loss_path, i):
    loss_data_path = loss_path + "/{}.mat".format(loss_name)
    losses = loadmat(loss_data_path)
    loss_train = losses["train"]
    # print(loss_train[0, :])
    # print("Test:", np.sum(loss_train[0, :])/len(loss_train[0, :]))
    loss_val = losses["val"]
    print("Loss Shape: Train:{}  Val:{}".format(loss_train.shape, loss_val.shape))
    average_loss_train = loss_train[:,loss_train.shape[1]-1]
    average_loss_val = loss_val[:,loss_val.shape[1]-1]
    # plt.figure()
    plt.plot(np.arange(loss_train.shape[0]), average_loss_train, color = "cyan", label="Train Average Loss")
    plt.plot(np.arange(loss_val.shape[0]), average_loss_val, color = "b", label="Val Average Loss")
    # plt.xlim([0, loss_train.shape[0]])
    if i == 0:
        plt.ylim(bottom=0, top=0.1)
    else:
        plt.ylim(bottom=0, top=0.005)
    plt.xlim([0, 100])
    plt.ylim(bottom=0)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Iter {}".format(i+1))
    # plt.show()
def main():
    if torch.cuda.is_available():
        root_path = "/mnt/tabgha/users/gy/MyProject"
    else:
        root_path = "V:/users/gy/MyProject"
    loss1_path = root_path + "/IterDa/results/Iter_1/v3/loss" ##V:\users\gy\MyProject\IterDa\results\Iter_1\v3\loss
    loss2_path = root_path + "/IterDa/results/Iter_2/v1/loss"
    loss3_path = root_path + "/IterDa/results/Iter_3/v1/loss"
    loss4_path = root_path + "/IterDa/results/Iter_4/v2/loss"
    # show_loss("losses", loss4_path)
    loss_path = [loss1_path, loss2_path, loss3_path, loss4_path]
    plt.figure()
    for i in range(4):
        plt.subplot(2,2,i+1)
        show_loss("losses", loss_path[i], i)
    plt.tight_layout()
    plt.savefig("V:/users/gy/毕业设计/生物医学工程/论文/图表/第四章/图4-4.png")
    plt.show()




if __name__ == "__main__":
    main()