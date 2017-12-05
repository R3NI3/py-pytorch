import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable, Function
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import math

has_cuda = torch.cuda.is_available()

class mdlstm(nn.Module):
    def __init__(self, input_size, hidden_size, img_dim):
        super(mdlstm, self).__init__()
        self.img_dim = img_dim
        self.mdlstm_dim = 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(self.mdlstm_dim, 4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(self.mdlstm_dim, 4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input):

        h_matrix = []
        s_matrix = []

        h_past = [0,0]
        s_past = [0,0]

        for pixel_idx,pixels in enumerate(torch.transpose(input,0,2)):

            pixels = torch.transpose(pixels,0,1)

            # calc of past dimensions for 1-d to 2-d
            #prof = idx // 1d_len*2d_len... new_idx = idx % 1d_len*2d_len...
            # TODO: Add generic calculation for d dimensions
            row = (pixel_idx // self.img_dim[1]) # idx divided column length]
            col =(pixel_idx % self.img_dim[1])

            if (row > 0):
                h_past[0] = h_matrix[(row-1)*self.img_dim[1] + col]
                s_past[0] = s_matrix[(row-1)*self.img_dim[1] + col]

            if (col > 0):
                h_past[1] = h_matrix[(row)*self.img_dim[1] + col-1]
                s_past[1] = s_matrix[(row)*self.img_dim[1] + col-1]

            for dim in range(self.mdlstm_dim):
                if (dim ==0 and row > 0):
                    hidden = F.linear(h_past[dim], self.weight_hh[dim], self.bias_hh[dim])
                elif(dim ==0):
                    if(has_cuda):
                        hidden = F.linear(Variable(torch.zeros(1,self.hidden_size)).cuda(), self.weight_hh[dim], self.bias_hh[dim])
                    else:
                        hidden = F.linear(Variable(torch.zeros(1,self.hidden_size)), self.weight_hh[dim], self.bias_hh[dim])
                if (dim ==1 and col > 0):
                    hidden = hidden + F.linear(h_past[dim], self.weight_hh[dim], self.bias_hh[dim])
                elif(dim == 1):
                    if(has_cuda):
                        hidden = hidden + F.linear(Variable(torch.zeros(1,self.hidden_size)).cuda(), self.weight_hh[dim], self.bias_hh[dim])
                    else:
                        hidden = hidden + F.linear(Variable(torch.zeros(1,self.hidden_size)), self.weight_hh[dim], self.bias_hh[dim])
            gates = (F.linear(pixels, self.weight_ih, self.bias_ih) + hidden)


            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            s_matrix.append(0)
            for dim in range(self.mdlstm_dim):
                if (dim ==0 and row > 0):
                    s_matrix[pixel_idx] = (forgetgate * s_past[dim])
                if (dim ==1 and col > 0):
                    s_matrix[pixel_idx] = s_matrix[pixel_idx] + (forgetgate * s_past[dim])

            s_matrix[pixel_idx] = s_matrix[pixel_idx] + (ingate * cellgate)
            h_matrix.append(outgate * F.tanh(s_matrix[pixel_idx]))

        h_tensor = (torch.stack(h_matrix,0))
        h_tensor = torch.transpose(h_tensor,0,1)
        h_tensor = torch.transpose(h_tensor,1,2).contiguous()
        return  h_tensor



