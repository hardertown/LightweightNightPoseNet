from nnet import Model
import torch


x = torch.randn(1, 1, 224, 224)
net = Model(num_classes=17)

torch.onnx.export(
    net,
    x,
    'model.onnx',
    export_params=True,
    opset_version=8,
)

