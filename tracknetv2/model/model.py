import torch


def create_conv_sub_layer(in_size, out_size):
    return torch.nn.Sequential(*[
        torch.nn.Conv2d(in_size, out_size, (3, 3), padding='same', stride=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(out_size, momentum=0.9, eps=1e-3)
    ])


def create_vgg_16_layer(in_size, out_size, num, pool):
    layers = []
    if pool:
        layers.append(torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
    layers.append(create_conv_sub_layer(in_size, out_size))
    for _ in range(num-1):
        layers.append(create_conv_sub_layer(out_size, out_size))
    return torch.nn.Sequential(*layers)


def create_deconv_layer(in_size, out_size, num):
    layers = []
    layers.append(create_conv_sub_layer(in_size, out_size))
    for _ in range(num-1):
        layers.append(create_conv_sub_layer(out_size, out_size))
    return torch.nn.Sequential(*layers)


class TrackNetV2(torch.nn.Module):
    def __init__(self, input_h: int, input_w: int, out: int = 3, dropout: float = 0.0):
        """
        TrackNetV2 model class.
        From "TrackNetV2: Efficient Shuttlecock Tracking Network" (2020) by Nien-En Sun, Yu-Ching Lin et al.
        :param input_h: Height of input images.
        :param input_w: Width of input images.
        :param out: Num of heatmaps to generate. Usually is the same as value of consecutive frames in your dataset.
        :param dropout: Probability of dropout.
        :return:
        """
        super().__init__()

        self.height = input_h
        self.width = input_w
        self.out = out

        self.dropout = torch.nn.Dropout(p=dropout) if dropout != 0 else None

        self.conv_1 = create_vgg_16_layer(in_size=3 * out, out_size=64, num=2, pool=False)
        self.conv_2 = create_vgg_16_layer(in_size=64, out_size=128, num=2, pool=True)
        self.conv_3 = create_vgg_16_layer(in_size=128, out_size=256, num=3, pool=True)
        self.conv_4 = create_vgg_16_layer(in_size=256, out_size=512, num=3, pool=True)

        self.upsampling = torch.nn.Upsample(scale_factor=2, mode='nearest')

        self.deconv_1 = create_deconv_layer(in_size=256+512, out_size=256, num=3)
        self.deconv_2 = create_deconv_layer(in_size=128+256, out_size=128, num=2)
        self.deconv_3 = create_deconv_layer(in_size=64+128, out_size=64, num=2)
        self.final_conv = torch.nn.Conv2d(64, out, (1, 1), padding='same', stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Produce heatmap.
        :param x: Images tensor of shape (N, C, H, W)
        :return: heatmap (N, out, H, W)
        """
        x1 = self.conv_1(x)
        if self.dropout is not None:
            x1 = self.dropout(x1)
        x2 = self.conv_2(x1)
        if self.dropout is not None:
            x2 = self.dropout(x2)
        x3 = self.conv_3(x2)
        if self.dropout is not None:
            x3 = self.dropout(x3)
        x = self.conv_4(x3)
        if self.dropout is not None:
            x = self.dropout(x)

        x = torch.cat([self.upsampling(x), x3], dim=1)
        x = self.deconv_1(x)
        x = torch.cat([self.upsampling(x), x2], dim=1)
        x = self.deconv_2(x)
        x = torch.cat([self.upsampling(x), x1], dim=1)
        x = self.deconv_3(x)
        x = self.final_conv(x)
        return x.sigmoid()
