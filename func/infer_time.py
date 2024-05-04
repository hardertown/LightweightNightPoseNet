import torch
import torch.nn as nn
# from model.nnet import Model
# from model.lnnet import Model
from model.lnnet_mobile_se import Model
import torch.utils.benchmark as benchmark




# 创建模型和输入数据
# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
input = torch.randn(64, 1, 224, 224).to(device)

# 进行推理速度测试
timer = benchmark.Timer(
    stmt='model(input)',
    setup='import torch',
    globals={'model': model, 'input': input}
)
time = timer.timeit(10).median  # 获取中位数

print(f"推理时间：{time} 秒")
