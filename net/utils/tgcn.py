# The based unit of graph convolutional networks.

import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)
        self.out_channels = out_channels
        self.conv_a = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_b = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        M = self.learnable_matrix(x, A)
        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A + M))

        return x.contiguous(), A

    def learnable_matrix(self, x, A):
        N, C, T, V = x.size()
        # print(A.size(), x.size())
        A1 = self.conv_a(x).permute(0,3,1,2).contiguous().view(N, V, self.out_channels*T)
        A2 = self.conv_b(x).view(N, self.out_channels*T, V)
        concat_matrix = torch.matmul(A1, A2) / A1.size(-1)
        result = torch.ones(A.size())

        if N >= 2:
            for i in range(A.size(0)):
                result[i] = concat_matrix[0]
        else:
            result = concat_matrix

        result.softmax(dim=1)
        return result.softmax(dim=2).cuda(x.get_device())
