import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):

        res = x #F.dropout(x, 0.3)
        res = self.body(res).mul(self.res_scale)
        res += x

        return res
        

class EDSR(nn.Module):
    def __init__(self, cin=1, n_resblocks=16, n_feats=64, res_scale=1):
        super(EDSR, self).__init__()

        conv = default_conv
        kernel_size = 3 
        act = nn.LeakyReLU(0.1, True) #nn.ReLU(True)

        # define head module
        m_head = [conv(cin, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            conv(n_feats, 1, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x 


class Crop2d(nn.Module):
    """Crop input using slicing. Assumes BCHW data.
    Args:
        crop (Tuple[int, int, int, int]): Amounts to crop from each side of the image.
            Tuple is treated as [left, right, top, bottom]/
    """

    def __init__(self, crop: Tuple[int, int, int, int]):
        super().__init__()
        self.crop = crop
        assert len(crop) == 4

    def forward(self, x: Tensor) -> Tensor:
        (left, right, top, bottom) = self.crop
        x0, x1 = left, x.shape[-1] - right
        y0, y1 = top, x.shape[-2] - bottom
        return x[:, :, y0:y1, x0:x1]


class Shift2d(nn.Module):
    """Shift an image in either or both of the vertical and horizontal axis by first
    zero padding on the opposite side that the image is shifting towards before
    cropping the side being shifted towards.
    Args:
        shift (Tuple[int, int]): Tuple of vertical and horizontal shift. Positive values
            shift towards right and bottom, negative values shift towards left and top.
    """

    def __init__(self, shift: Tuple[int, int]):
        super().__init__()
        self.shift = shift
        vert, horz = self.shift
        y_a, y_b = abs(vert), 0
        x_a, x_b = abs(horz), 0
        if vert < 0:
            y_a, y_b = y_b, y_a
        if horz < 0:
            x_a, x_b = x_b, x_a
        # Order : Left, Right, Top Bottom
        self.pad = nn.ZeroPad2d((x_a, x_b, y_a, y_b))
        self.crop = Crop2d((x_b, x_a, y_b, y_a))
        self.shift_block = nn.Sequential(self.pad, self.crop)

    def forward(self, x: Tensor) -> Tensor:
        return self.shift_block(x)
        

def rotate(x: torch.Tensor, angle: int) -> torch.Tensor:
    """Rotate images by 90 degrees clockwise. Can handle any 2D data format.
    Args:
        x (Tensor): Image or batch of images.
        angle (int): Clockwise rotation angle in multiples of 90.
        data_format (str, optional): Format of input image data, e.g. BCHW,
            HWC. Defaults to BCHW.
    Returns:
        Tensor: Copy of tensor with rotation applied.
    """
    h_dim = 2
    w_dim = 3

    if angle == 0:
        return x
    elif angle == 90:
        return x.flip(w_dim).transpose(h_dim, w_dim)
    elif angle == 180:
        return x.flip(w_dim).flip(h_dim)
    elif angle == 270:
        return x.flip(h_dim).transpose(h_dim, w_dim)
    else:
        raise NotImplementedError("Must be rotation divisible by 90 degrees")


class NoiseNetwork(nn.Module):
    """Custom U-Net architecture for Self Supervised Denoising (SSDN) and Noise2Noise (N2N).
    Base N2N implementation was made with reference to @joeylitalien's N2N implementation.
    Changes made are removal of weight sharing when blocks are reused. Usage of LeakyReLu
    over standard ReLu and incorporation of blindspot functionality.
    Unlike other typical U-Net implementations dropout is not used when the model is trained.
    When in blindspot mode the following behaviour changes occur:
        * Input batches are duplicated for rotations: 0, 90, 180, 270. This increases the
          batch size by 4x. After the encode-decode stage the rotations are undone and
          concatenated on the channel axis with the associated original image. This 4x
          increase in channel count is collapsed to the standard channel count in the
          first 1x1 kernel convolution.
        * To restrict the receptive field into the upward direction a shift is used for
          convolutions (see ShiftConv2d) and downsampling. Downsampling uses a single
          pixel shift prior to max pooling as dictated by Laine et al. This is equivalent
           to applying a shift on the upsample.
    Args:
        in_channels (int, optional): Number of input channels, this will typically be either
            1 (Mono) or 3 (RGB) but can be more. Defaults to 3.
        out_channels (int, optional): Number of channels the final convolution should output.
            Defaults to 3.
        blindspot (bool, optional): Whether to enable the network blindspot. This will
            add in rotation stages and shift stages while max pooling and during convolutions.
            A futher shift will occur after upsample. Defaults to False.
        zero_output_weights (bool, optional): Whether to initialise the weights of
                `nin_c` to zero. This is not mentioned in literature but is done as part
                of the tensorflow implementation for the parameter estimation network.
                Defaults to False.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        blindspot: bool = False,
        zero_output_weights: bool = False,
    ):
        super(NoiseNetwork, self).__init__()
        self._blindspot = blindspot
        self._zero_output_weights = zero_output_weights
        self.Conv2d = ShiftConv2d if self.blindspot else nn.Conv2d

        ####################################
        # Encode Blocks
        ####################################

        def _max_pool_block(max_pool: nn.Module) -> nn.Module:
            if blindspot:
                return nn.Sequential(Shift2d((1, 0)), max_pool)
            return max_pool

        # Layers: enc_conv0, enc_conv1, pool1
        self.encode_block_1 = nn.Sequential(
            self.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            self.Conv2d(48, 48, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            _max_pool_block(nn.MaxPool2d(2)),
        )

        # Layers: enc_conv(i), pool(i); i=2..5
        def _encode_block_2_3_4_5() -> nn.Module:
            return nn.Sequential(
                #nn.Dropout(p_drop),  ####
                self.Conv2d(48, 48, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                _max_pool_block(nn.MaxPool2d(2)),
            )

        # Separate instances of same encode module definition created
        self.encode_block_2 = _encode_block_2_3_4_5()
        self.encode_block_3 = _encode_block_2_3_4_5()
        self.encode_block_4 = _encode_block_2_3_4_5()
        self.encode_block_5 = _encode_block_2_3_4_5()

        # Layers: enc_conv6
        self.encode_block_6 = nn.Sequential(
            #nn.Dropout(p_drop),  ####
            self.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        ####################################
        # Decode Blocks
        ####################################
        # Layers: upsample5
        self.decode_block_6 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self.decode_block_5 = nn.Sequential(
            #nn.Dropout(p_drop),  ####
            self.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            #nn.Dropout(p_drop),  ####
            self.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        def _decode_block_4_3_2() -> nn.Module:
            return nn.Sequential(
                #nn.Dropout(p_drop),  ####
                self.Conv2d(144, 96, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                #nn.Dropout(p_drop),  ####
                self.Conv2d(96, 96, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )

        # Separate instances of same decode module definition created
        self.decode_block_4 = _decode_block_4_3_2()
        self.decode_block_3 = _decode_block_4_3_2()
        self.decode_block_2 = _decode_block_4_3_2()

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self.decode_block_1 = nn.Sequential(
            #nn.Dropout(p_drop),  ####
            self.Conv2d(96 + in_channels, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            #nn.Dropout(p_drop),  ####
            self.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        ####################################
        # Output Block
        ####################################

        if self.blindspot:
            # Shift 1 pixel down
            self.shift = Shift2d((1, 0))
            # 4 x Channels due to batch rotations
            nin_a_io = 384
        else:
            nin_a_io = 96

        # nin_a,b,c, linear_act
        self.output_conv = self.Conv2d(96, out_channels, 1)
        self.output_block = nn.Sequential(
            #nn.Dropout(p_drop),  ####
            self.Conv2d(nin_a_io, nin_a_io, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            #nn.Dropout(p_drop),  ####
            self.Conv2d(nin_a_io, 96, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            self.output_conv,
        )

        # Initialize weights
        #self.init_weights()

    @property
    def blindspot(self) -> bool:
        return self._blindspot

    def init_weights(self):
        """Initializes weights using Kaiming  He et al. (2015).
        Only convolution layers have learnable weights. All convolutions use a leaky
        relu activation function (negative_slope = 0.1) except the last which is just
        a linear output.
        """
        with torch.no_grad():
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0.1)
                m.bias.data.zero_()
        # Initialise last output layer
        if self._zero_output_weights:
            self.output_conv.weight.zero_()
        else:
            nn.init.kaiming_normal_(self.output_conv.weight.data, nonlinearity="linear")

    def forward(self, x: Tensor) -> Tensor:
        if self.blindspot:
            rotated = [rotate(x, rot) for rot in (0, 90, 180, 270)]
            x = torch.cat((rotated), dim=0)

        # Encoder
        pool1 = self.encode_block_1(x)
        pool2 = self.encode_block_2(pool1)
        pool3 = self.encode_block_3(pool2)
        pool4 = self.encode_block_4(pool3)
        pool5 = self.encode_block_5(pool4)
        encoded = self.encode_block_6(pool5)

        # Decoder
        upsample5 = self.decode_block_6(encoded)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.decode_block_5(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.decode_block_4(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.decode_block_3(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.decode_block_2(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        x = self.decode_block_1(concat1)

        # Output
        if self.blindspot:
            # Apply shift
            shifted = self.shift(x)
            # Unstack, rotate and combine
            rotated_batch = torch.chunk(shifted, 4, dim=0)
            aligned = [
                rotate(rotated, rot)
                for rotated, rot in zip(rotated_batch, (0, 270, 180, 90))
            ]
            x = torch.cat(aligned, dim=1)

        x = self.output_block(x)

        return x

    @staticmethod
    def input_wh_mul() -> int:
        """Multiple that both the width and height dimensions of an input must be to be
        processed by the network. This is devised from the number of pooling layers that
        reduce the input size.
        Returns:
            int: Dimension multiplier
        """
        max_pool_layers = 5
        return 2 ** max_pool_layers


class ShiftConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shift_size = (self.kernel_size[0] // 2, 0)
        # Use individual layers of shift for wrapping conv with shift
        shift = Shift2d(self.shift_size)
        self.pad = shift.pad
        self.crop = shift.crop

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(x)
        x = super().forward(x)
        x = self.crop(x)
        return x