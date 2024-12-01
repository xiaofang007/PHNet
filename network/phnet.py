from torch import nn
from .MLPP import MLPP
from .basic_block import conv_layer, deconv_layer


class PHNet(nn.Module):
    def __init__(self, res_ratio, layers, in_channels, out_channels, embed_dims, segment_dim, mlp_ratio, dropout_rate):
        super(PHNet,self).__init__()
        conv_dim = [in_channels,embed_dims[0],embed_dims[1],embed_dims[2]]
        self.conv = conv_layer(conv_dim, res_ratio, dropout_rate=dropout_rate)
        self.deconv = deconv_layer(embed_dims, res_ratio,dropout_rate=dropout_rate)
        self.mlpp = MLPP(res_ratio,layers,in_channels=embed_dims[-3],embed_dims=embed_dims[-2:],segment_dim=segment_dim,
                                    mlp_ratios=mlp_ratio,dropout_rate=dropout_rate)
        self.final_conv = nn.Conv3d(embed_dims[0],out_channels=out_channels,kernel_size=1)

    def forward(self, x):
        conv_hidden_states = self.conv(x)
        mlpp_hidden_states = self.mlpp(conv_hidden_states[-1])
        hidden_states = (conv_hidden_states + mlpp_hidden_states)[::-1]
        u0 = self.deconv(hidden_states)
        logits = self.final_conv(u0)
        return logits
