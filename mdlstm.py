import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable, Function
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class mdlstm(nn.Module):
    def __init__(self, input_size, hidden_size, img_dim):
        super(mdlstm, self).__init__()
        self.img_dim = img_dim
        self.mdlstm_dim = 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, 1))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.Tensor(4 * hidden_size)) # apenas 1 bias msm?

    def forward(self, input):
        h_matrix = []
        s_matrix = []

        h_past = [0,0]
        s_past = [0,0]
        for pixel_idx,pixels in enumerate(input.t()):
            pixels = pixels.contiguous().view(-1,1)
            # calc of past dimensions for 1-d to 2-d
            #prof = idx // 1d_len*2d_len... new_idx = idx % 1d_len*2d_len...
            # TODO: Add generic calculation for d dimensions
            row = (pixel_idx // self.img_dim[1]) # idx divided column length
            col =(pixel_idx % self.img_dim[1])

            if (row > 0):
                h_past[0] = h_matrix[(row-1)*self.img_dim[1] + col]
                s_past[0] = s_matrix[(row-1)*self.img_dim[1] + col]

            if (col > 0):
                h_past[1] = h_matrix[(row)*self.img_dim[1] + col-1]
                s_past[1] = s_matrix[(row)*self.img_dim[1] + col-1]

            hidden = 0
            for dim in range(self.mdlstm_dim):
                if (dim ==0 and row > 0):
                    hidden = hidden + F.linear(h_past[dim], self.weight_hh)
                if (dim ==1 and col > 0):
                    hidden = hidden + F.linear(h_past[dim], self.weight_hh)
            gates = (F.linear(pixels, self.weight_ih) + hidden + self.bias)

            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)
            s_matrix.append(0)
            for dim in range(self.mdlstm_dim):
                if (dim ==0 and row > 0):
                    s_matrix[pixel_idx] = s_matrix[pixel_idx] + (forgetgate * s_past[dim])
                if (dim ==1 and col > 0):
                    s_matrix[pixel_idx] = s_matrix[pixel_idx] + (forgetgate * s_past[dim])
            s_matrix[pixel_idx] = s_matrix[pixel_idx] + (ingate * cellgate)
            h_matrix.append(outgate * F.tanh(s_matrix[pixel_idx]))

        h_tensor = torch.cat(h_matrix,1)
        return  h_tensor
        # result will be NxHxI (Ex: mnist batch =100, 128 hidden cells -> 100x128x784)


