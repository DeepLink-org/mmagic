from mmengine import MODELS, Config
from torchvision import utils

from mmengine.registry import init_default_scope

import pickle as pkl
import os

from collections import OrderedDict



storage_path = os.getenv('ONE_ITER_TOOL_STORAGE_PATH', 'one_iter_data')
if not os.path.exists(storage_path):
    os.makedirs(storage_path)

devices = ['cpu', 'dipu']
data_paths = []
for device in devices:

    print("stable diffusion ",device," begin")

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

    image = StableDiffuser.infer(prompt)['samples'][0]
    image.save(f'robot_{device}.png')

    if iout_dict:
        with open(io_path, "wb") as f: 
            pkl.dump(iout_dict, f)

    data_paths.append(io_path)

    print("stable diffusion ",device," pass")

os.environ["ONE_ITER_TOOL_DEVICE"] = devices[0]
os.environ["ONE_ITER_TOOL_DEVICE_COMPARE"] = devices[1]
from cp import compare
compare(data_paths[0], data_paths[1])
print('stable diffusion successfully pass the all of the tests!')