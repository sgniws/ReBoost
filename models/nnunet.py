from torch import nn
from torchsummary import summary
import torch


# This 3D-UNet structure is from nnUNet's configuration for BraTS2018 task. Thanks, nnUNet authors!
class UNet(nn.Module):
    def __init__(self, input_channels,
                 n_classes, n_stages, n_features_per_stage, kernel_size, strides,
                 apply_deep_supervision=False):
        super().__init__()

        self.input_channels = input_channels
        self.n_classes = n_classes
        self.n_stages = n_stages
        self.n_features_per_stage = n_features_per_stage
        self.kernel_size = kernel_size
        self.strides =strides
        self.apply_deep_supervision = apply_deep_supervision

        self._build_encoder()
        self._build_decoder()

    def _build_encoder(self):
        stages = []

        for i in range(self.n_stages):
            cur_stage = [
                nn.Conv3d(
                    in_channels=self.input_channels if i == 0 else self.n_features_per_stage[i - 1],
                    out_channels=self.n_features_per_stage[i],
                    kernel_size=self.kernel_size[i],
                    stride=self.strides[i],
                    padding=[(_ - 1) // 2 for _ in self.kernel_size[i]],
                    dilation=1,
                    bias=True,
                ),
                nn.modules.instancenorm.InstanceNorm3d(self.n_features_per_stage[i], eps=1e-05, affine=True),
                nn.modules.activation.LeakyReLU(inplace=True),

                nn.Conv3d(
                    in_channels=self.n_features_per_stage[i],
                    out_channels=self.n_features_per_stage[i],
                    kernel_size=self.kernel_size[i],
                    stride=1,
                    padding=[(_ - 1) // 2 for _ in self.kernel_size[i]],
                    dilation=1,
                    bias=True,
                ),
                nn.modules.instancenorm.InstanceNorm3d(self.n_features_per_stage[i], eps=1e-05, affine=True),
                nn.modules.activation.LeakyReLU(inplace=True),
            ]

            stages.append(nn.Sequential(*cur_stage))

        self.encoder_stages = nn.ModuleList(stages)

    def _build_decoder(self):
        connect_layers = []
        stages = []
        seg_layers = []

        for i in range(1, self.n_stages):
            connect_layers.append(
                nn.ConvTranspose3d(
                    in_channels=self.n_features_per_stage[-i],
                    out_channels=self.n_features_per_stage[-i - 1],
                    kernel_size=self.strides[-i],
                    stride=self.strides[-i],
                    bias=True
                )
            )

            cur_stage = [
                nn.ConvTranspose3d(
                    in_channels=2 * self.n_features_per_stage[-i - 1],
                    out_channels=self.n_features_per_stage[-i - 1],
                    kernel_size=self.kernel_size[-i - 1],
                    stride=1,
                    padding=[(_ - 1) // 2 for _ in self.kernel_size[-i - 1]],
                    dilation=1,
                    bias=True,
                ),
                nn.modules.instancenorm.InstanceNorm3d(self.n_features_per_stage[-i - 1], eps=1e-05, affine=True),
                nn.modules.activation.LeakyReLU(inplace=True),

                nn.ConvTranspose3d(
                    in_channels=self.n_features_per_stage[-i - 1],
                    out_channels=self.n_features_per_stage[-i - 1],
                    kernel_size=self.kernel_size[-i - 1],
                    stride=1,
                    padding=[(_ - 1) // 2 for _ in self.kernel_size[-i - 1]],
                    dilation=1,
                    bias=True,
                ),
                nn.modules.instancenorm.InstanceNorm3d(self.n_features_per_stage[-i - 1], eps=1e-05, affine=True),
                nn.modules.activation.LeakyReLU(inplace=True),
            ]

            stages.append(nn.Sequential(*cur_stage))

            seg_layers.append(
                nn.Conv3d(
                    in_channels=self.n_features_per_stage[-i - 1],
                    out_channels=self.n_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            )

        self.connect_layers = nn.ModuleList(connect_layers)
        self.decoder_stages = nn.ModuleList(stages)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, x):
        encoded_feat_maps = []
        for stage in self.encoder_stages:
            x = stage(x)
            encoded_feat_maps.append(x)

        low_res_input = encoded_feat_maps[-1]
        outputs = []
        for i in range(len(self.decoder_stages)):
            output = self.connect_layers[i](low_res_input)
            output = torch.cat((output, encoded_feat_maps[-i - 2]), dim=1)
            output = self.decoder_stages[i](output)
            low_res_input = output

            output = self.seg_layers[i](output)

            if self.apply_deep_supervision:
                outputs.append(output)
            elif i == len(self.decoder_stages) - 1:
                outputs.append(output)

        if self.apply_deep_supervision:
            outputs = outputs[:: -1]
        else:
            outputs = outputs[0]

        return outputs


if __name__ == '__main__':
    net = UNet(
        input_channels=4,
        n_classes=3,
        n_stages=6,
        n_features_per_stage=[32, 64, 128, 256, 320, 320],
        kernel_size=[[3, 3, 3]] * 6,
        strides=[[1, 1, 1], * [[2, 2, 2]] * 5],
        apply_deep_supervision=False
    ).cuda()
    summary(net, (4, 128, 128, 128))
    print(f'Output shape give input shape (2, 4, 128, 128, 128): {net(torch.rand(2, 4, 128, 128, 128).cuda()).shape}')
