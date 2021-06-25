import torch
import torch.nn as nn


class PixelDiscriminator(nn.Module):
    def __init__(self, in_channels=49, out_channels=1):
        super(PixelDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=False, dropout=0.0):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1, bias=False)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout:
                layers.append(nn.Dropout(dropout))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(45, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            *discriminator_block(512, 1024),
            *discriminator_block(1024, 2048),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(2048, out_channels, 4, padding=1, stride=2, bias=False),
            # nn.Softmax(dim=1)

        )
        self.conv1 = nn.Sequential(nn.Conv2d(20, 30, (3, 3), padding=1),
                                   nn.Conv2d(30, 30, (2, 2), padding=1, dilation=2),
                                   nn.Conv2d(30, 30, (1, 1)))

    def forward(self, img_A, img_B=None):
        # img_A = img_A.half()
        # img_B = img_B.half()
        img_B = self.conv1(img_B)
        # img_B = img_B.view(img_A.shape[0], -1, img_A.shape[2], img_A.shape[3])
        img_B = torch.nn.Upsample(size=(img_A.shape[2], img_A.shape[3]), mode='bilinear', align_corners=True)(img_B)

        # img_B = img_B.view(img_A.shape[0],-1,img_A.shape[2],img_A.shape[3])
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        output = self.model(img_input)
        # output /= 0.005
        # softmax = nn.Softmax(dim=1)(output)
        return output  # ,softmax #self.model(img_input)#.view(img_A.shape[0], -1)
