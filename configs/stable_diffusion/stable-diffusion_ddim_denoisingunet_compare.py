import os

os.environ["ONE_ITER_TOOL_DEVICE"] = 'dipu'
os.environ["ONE_ITER_TOOL_DEVICE_COMPARE"] = 'cpu'

storage_path = os.getenv('ONE_ITER_TOOL_STORAGE_PATH', None)
assert storage_path is not None, "Environment variable `ONE_ITER_TOOL_STORAGE_PATH` not found, please set first"


data_paths_0 = f'{storage_path}/data_dipu.pth'
data_paths_1 = f'{storage_path}/baseline/data_cpu.pth'

from cp import compare
compare(data_paths_0, data_paths_1, compare_type='forward')
print('stable diffusion successfully pass the all of the tests!')