from mmengine import MODELS, Config
from torchvision import utils

from mmengine.registry import init_default_scope
import inspect

import pickle as pkl
import os

from collections import OrderedDict
import sys
import numpy as np
import torch

np.random.seed(10086)
torch.manual_seed(10086)
torch.cuda.manual_seed_all(10086)

device =  sys.argv[1]

io_save_ratio = 0.2

if device != 'cpu' and device != 'dipu':
    print("error device :{}".format(device), flush = True)
    exit(1)

storage_path = os.getenv('ONE_ITER_TOOL_STORAGE_PATH', None)
assert storage_path is not None, "Environment variable `ONE_ITER_TOOL_STORAGE_PATH` not found, please set first"

if not os.path.exists(storage_path):
    os.makedirs(storage_path)

storage_baseline_path = f'{storage_path}/baseline'

#如果未正确配置软链接，则直接将基准数据存到本地
if not os.path.exists(storage_baseline_path):
    os.makedirs(storage_baseline_path)


data_paths = []

if device == "cpu":
    storage_path = storage_baseline_path

print("stable diffusion ",device," begin")

if device == 'cpu' and os.path.islink(storage_baseline_path):
    io_save_ratio = 1.0
    print("Save cpu data in {}, which links to common path, so we save it all!".format(storage_baseline_path), flush = True)

os.environ["ONE_ITER_TOOL_DEVICE"] = device 
os.environ["ONE_ITER_TOOL_MODE"] = "others"
if (device=='cpu'):
    os.environ["DIPU_MOCK_CUDA"] = "False"
else:
    os.environ["DIPU_MOCK_CUDA"] = "True"
import capture
io_path = f'{storage_path}/data_{device}.pth'
if device == 'dipu':
    import torch_dipu

init_default_scope('mmagic')

config = 'mmagic/configs/stable_diffusion/stable-diffusion_ddim_denoisingunet.py'
config = Config.fromfile(config).copy()

StableDiffuser = MODELS.build(config.model)
prompt = 'A mecha robot in a favela in expressionist style'
StableDiffuser = StableDiffuser.to('cuda')

iout_dict = OrderedDict()
capture.register_hook(StableDiffuser, iout_dict)


image = StableDiffuser.infer(prompt,num_inference_steps=1)['samples'][0]
image.save(f'robot_{device}.png')


if iout_dict:
    delete_ratio = 1-float(io_save_ratio)
    iout_dict_keys = list(iout_dict.keys())
    ordered_dict_keys = []  #存储过滤出的key
    for key in iout_dict_keys:
        if "data_preprocessor" in key:
            del iout_dict[key] #不保存包含data_preprocessor字段的key值
        elif "loss" not in key:
            ordered_dict_keys.append(key) # 保留包含"loss"键
    if delete_ratio >= (float(len(ordered_dict_keys))/float(len(iout_dict))): #如果删除的比例大于预删除的大小,那就只保留loss字段的key,因为loss是重要的比较字段
        for key in ordered_dict_keys:
            del iout_dict[key] 
    else:
        random_indices = np.random.choice(len(ordered_dict_keys), size=int(len(iout_dict) * delete_ratio), replace=False)  # 否则，随机获取要删除键的位置
        for index in random_indices:
            del iout_dict[ordered_dict_keys[index]]
    with open(io_path, "wb") as f:
        pkl.dump(iout_dict, f)



print("stable diffusion {} pass".format(device), flush = True)
