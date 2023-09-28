from torch import nn

class DilationConv(nn.Sequential):
    def __init__(self, in_channels, out_channels,k_size=3, dilation=1,padding=0):
        modules = [
            nn.Conv1d(in_channels, out_channels, k_size, padding=padding, dilation=dilation, bias=False),
        ]
        super(DilationConv, self).__init__(*modules)

class DSConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,dilation=1,padding=0,bias=True):
        super(DSConv,self).__init__()
        self.body = nn.Sequential(
                        nn.Conv2d(in_channels = in_channels, out_channels = in_channels,
                                              kernel_size = (kernel_size, 1),
                                              stride = stride,
                                              padding = (padding,0), dilation = dilation, groups = in_channels, bias = bias),
                        # 1x3
                        nn.Conv2d(in_channels = in_channels, out_channels = in_channels,
                                              kernel_size = (1, kernel_size),
                                              stride = stride,
                                              padding = (0,padding) , dilation = dilation, groups = in_channels, bias = bias),
                        # PointWise Conv
                        nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = bias)
                    )

    def forward(self,x):
        return self.body(x)