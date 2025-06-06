import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvINReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, p=0.2):
        super(_ConvINReLU3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(out_channels),
            nn.Dropout3d(p=p, inplace=True),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _ConvIN3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(_ConvIN3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(out_channels)
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_chns, out_chns, k=1, p=0, dropout=0.2):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            _ConvINReLU3D(in_channels=in_chns,out_channels=out_chns,kernel_size=k,padding=p,p=dropout),
            _ConvIN3D(in_channels=out_chns,out_channels=out_chns,kernel_size=k,padding=p),
        )
        self.conv2 = nn.Sequential(
            _ConvINReLU3D(in_channels=out_chns, out_channels=out_chns, kernel_size=k, padding=p,p=dropout),
            _ConvIN3D(in_channels=out_chns, out_channels=out_chns, kernel_size=k, padding=p),
        )
        self.conv3 = nn.Conv3d(in_channels=in_chns, out_channels=out_chns, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.conv3(x)
        x1 = self.relu(x1+x)
        x2 = self.conv2(x1)
        x2 = self.relu(x2+x1)
        return x2



class Encoder_s(nn.Module):
    def __init__(self, in_chns, out_chns, k=1, p=0, dropout=0.2):
        super(Encoder_s, self).__init__()
        self.conv1 = nn.Sequential(
            _ConvINReLU3D(in_channels=in_chns,out_channels=out_chns,kernel_size=k,padding=p,p=dropout),
            _ConvIN3D(in_channels=out_chns,out_channels=out_chns,kernel_size=k,padding=p),
        )
        self.conv2 = nn.Conv3d(in_channels=in_chns, out_channels=out_chns, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.conv2(x)
        x1 = self.relu(x1+x)
        return x1


class Decoder(nn.Module):  # learn from Efficient Context-Aware Network
    def __init__(self, in_chns,out_chns,dropout):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(   # 3x3x3 is implemented in 3x3x1->1x1x3 for anisotropic kernel
             _ConvINReLU3D(in_channels=in_chns,out_channels=out_chns,kernel_size=(3,3,1),padding=(1,1,0),p=dropout),
             _ConvIN3D(in_channels=out_chns, out_channels=out_chns, kernel_size=(1,1,3), padding=(0,0,1)),
        )
        self.conv2 = nn.Conv3d(in_channels=in_chns,out_channels=out_chns,kernel_size=1,padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.conv2(x)
        x1 = self.relu(x1 + x)  # residual block
        return x1


class Up_cat(nn.Module):  # kernel and stride is for deconvolution
    def __init__(self, in_chns, cat_chns, out_chns, kernel,stride,dropout,attention_block=None, halves=True):
        super(Up_cat, self).__init__()
        up_chns = in_chns//2 if halves else in_chns
        self.up = nn.ConvTranspose3d(in_chns,up_chns,kernel_size=kernel,stride=stride)
        self.attention = attention_block
        self.convs = Decoder(cat_chns+up_chns ,out_chns,dropout)

    def forward(self,x1,x2):   # x1 is to perform upsampling, x2 is from the encoder block
        x_1 = self.up(x1)
        # handling spatial shapes due to the 2x max pooling with odd edge lengths.
        if x2 is not None:
            dimensions = len(x1.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x2.shape[-i - 1] != x_1.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_1 = F.pad(x_1, sp, "replicate")
            x = torch.cat([x2,x_1],dim=1)   # concatenation or summation?
            if self.attention is not None:   # if there is an attention
                x,w = self.attention(x)
                x = self.convs(x)
                return x,w
            else:
                x = self.convs(x)
                return x

        else:
            x = self.convs(x_1)
            return x


class Up_sum(nn.Module):  # kernel and stride is for deconvolution
    def __init__(self, in_chns, out_chns, kernel,stride,dropout,attention_block=None, halves=True):
        super(Up_sum, self).__init__()
        up_chns = in_chns//2 if halves else in_chns
        self.up = nn.ConvTranspose3d(in_chns,up_chns,kernel_size=kernel,stride=stride)
        self.attention = attention_block
        self.convs = Decoder(up_chns ,out_chns,dropout)

    def forward(self,x1,x2):   # x1 is to perform upsampling, x2 is from the encoder block
        x_1 = self.up(x1)
        # handling spatial shapes due to the 2x max pooling with odd edge lengths.
        if x2 is not None:
            dimensions = len(x1.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x2.shape[-i - 1] != x_1.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            if sp != [0] * (dimensions * 2):
                x_1 = F.pad(x_1, sp, "replicate")
            x = x_1+x2  # concatenation or summation?
            if self.attention is not None:   # if there is an attention
                x,w = self.attention(x)
                x = self.convs(x)
                return x,w
            else:
                x = self.convs(x)
                return x

        else:
            x = self.convs(x)
            return x


class conv_layer(nn.Module):
    def __init__ (self, dim, res_ratio, dropout_rate):
        super(conv_layer, self).__init__()
        self.network = []
        for i in range(3):
            if res_ratio < 2**i+ 2**(i-1):  # isotropic
                self.network.append(Encoder(in_chns=dim[i],out_chns=dim[i+1],k=3,p=1, dropout=dropout_rate))
                if i < 2:
                    self.network.append(nn.MaxPool3d(kernel_size=2,stride=2))
            else:  # anisotropic
                self.network.append(Encoder(in_chns=dim[i],out_chns=dim[i+1],k=(3,3,1),p=(1,1,0), dropout=dropout_rate))
                if i < 2:
                    self.network.append(nn.MaxPool3d(kernel_size=(2,2,1),stride=(2,2,1)))
        
        self.network = nn.Sequential(*self.network)
        
    def forward(self,x):
        output = []
        for i in range(len(self.network)):
            x = self.network[i](x)
            if i % 2 ==0:
                output.append(x)
        
        return output


class deconv_layer(nn.Module):
    def __init__ (self, embed_dims, res_ratio, dropout_rate):
        super(deconv_layer, self).__init__()
        self.network = []
        for i in range(len(embed_dims)-1, 0, -1):
            is_half = True if embed_dims[i] == 2*embed_dims[i-1] else False
            if i <= 3 and res_ratio >= 2**(i-1)+ 2**(i-2): # anisotropic 
                self.network.append(Up_sum(in_chns=embed_dims[i], out_chns=embed_dims[i-1], kernel=(2, 2, 1), stride=(2, 2, 1), dropout=dropout_rate,
                                    halves=is_half))
            else:
                self.network.append(Up_sum(in_chns=embed_dims[i], out_chns=embed_dims[i-1], kernel=2, stride=2, dropout=dropout_rate,
                    halves=is_half))
        self.network = nn.Sequential(*self.network)   
    
    def forward(self, hidden_states):
        for i in range(0, len(self.network)):
            if i == 0:
                x = self.network[i](hidden_states[0], hidden_states[1])
            else:
                x = self.network[i](x, hidden_states[i+1])
        
        return x

