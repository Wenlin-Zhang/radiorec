
import torch
import timm


def replace_conv(parent):
    for n, m in parent.named_children():
        if type(m) is torch.nn.Conv2d:
            setattr(
                    parent,
                    n,
                    torch.nn.Conv1d(
                        m.in_channels,
                        m.out_channels,
                        kernel_size=m.kernel_size[0],
                        stride=m.stride[0],
                        padding=m.padding[0],
                        bias=m.kernel_size[0],
                        groups=m.groups,
                    ),
                )
        else:
            replace_conv(m)


def replace_bn(parent):
    for n, m in parent.named_children():
        if type(m) is torch.nn.BatchNorm2d:
            setattr(
                    parent,
                    n,
                    torch.nn.BatchNorm1d(
                        num_features = m.num_features
                    ),
                )
        else:
            replace_bn(m)


def replace_mp(parent):
    for n, m in parent.named_children():
        if type(m) is torch.nn.MaxPool2d:
            setattr(
                    parent,
                    n,
                    torch.nn.MaxPool1d(
                        kernel_size = m.kernel_size,
                        stride = m.stride,
                        padding = m.padding,
                        dilation = m.dilation
                    ),
                )
        else:
            replace_mp(m)


class FastAdaptiveMaxPool(torch.nn.Module):
    def __init__(self, flatten: bool = True):
        super(FastAdaptiveMaxPool, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        return x.amax(-1, keepdim=not self.flatten)


def create_resnet(num_classes, in_chans, name='resnet18', pretrained=False):
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)
    replace_conv(model)
    replace_bn(model)
    replace_mp(model)
    model.global_pool = FastAdaptiveMaxPool()
    return model
