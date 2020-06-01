import torch.nn.functional as F
from model_function import *
from torch import nn

class ResUnet(nn.Module):
    def __init__(self, args):
        super(ResUnet, self).__init__()
        self.input_channels = args.input_channels
        self.output_channels = args.output_channels
        self.bilinear = args.bilinear
        self.k_size = args.k_size

        self.res1 = ResBasic(self.input_channels, 64, k_size=self.k_size)
        self.res2 = ResBasic(64, 128, k_size=self.k_size)
        self.res3 = ResBasic(128, 256, k_size=self.k_size)
        self.res4 = ResBasic(256, 256, k_size=self.k_size)

        self.cat_res1 = CatRes(512, 128, k_size=self.k_size)
        self.cat_res2 = CatRes(256, 64, k_size=self.k_size)
        self.cat_res3 = CatRes(128, 64, k_size=self.k_size)
        self.out_lay = OutConv(64, self.output_channels, k_size=self.k_size)

    def forward(self, x):
        x1 = self.res1(x)
        x2 = self.res2(x1)
        x3 = self.res3(x2)
        x4 = self.res4(x3)

        x = self.cat_res1(x4, x3)
        x = self.cat_res2(x, x2)
        x = self.cat_res3(x, x1)
        result = self.out_lay(x)
        return result


class Stack_Bi_LSTM(nn.Module):
    def __init__(self,args):
        super(Stack_Bi_LSTM, self).__init__()
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.bias = args.bias
        self.batch_first = args.batch_first
        self.dropout = args.dropout
        self.bidirectional = args.bidirectional

        self.fc_hidden_size = args.fc_hidden_size
        self.output_size = args.output_size

        self.input_channels = 1
        self.output_channels = 1
        self.bilinear = True

        self.res1 = ResBasic(self.input_channels, 64)
        self.res2 = ResBasic(64, 64)
        # self.res3 = ResBasic(128, 256)
        # self.res4 = ResBasic(256, 256)

        self.cat_res1 = CatRes(512, 128)
        self.cat_res2 = CatRes(256, 64)
        self.cat_res3 = CatRes(128, 64)
        self.out_lay = OutConv(64, self.output_channels)

        if self.bidirectional:
            coefficient = 2
        else:
            coefficient = 1


        """
        LSTM:

        input(seq_len, batch, input_size)
        h_0(num_layers * num_directions, batch, hidden_size)
        c_0(num_layers * num_directions, batch, hidden_size)

        output(seq_len, batch, hidden_size * num_directions)
        h_n(num_layers * num_directions, batch, hidden_size)
        c_n(num_layers * num_directions, batch, hidden_size)
        """

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bias=self.bias,
                            batch_first=self.batch_first,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)
        # self.fc1 = nn.Linear(in_features = args.full_view * self.hidden_size * coefficient, out_features = args.full_view * args.fc_hidden_size)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(in_features = (args.full_view, args.fc_hidden_size), out_features = (args.full_view, args.output_size))


    def forward(self,x):
        x, (_,_) = self.lstm(x)

        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # x1 = self.res1(x.unsqueeze_(0).permute(1,0,2,3))
        # print(result.squeeze(1) .shape)
        # return result[:, 0, :, :]
        # return result.squeeze(1) 
        # result = x[:,:,:self.hidden_size] + x[:,:,self.hidden_size : self.hidden_size * 2]
        return x




