set -e

python mmagic/configs/stable_diffusion/stable-diffusion_ddim_denoisingunet_infer.py cpu

export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH  #如果使用dipu运算，将dipu纳入pythonpath里，使之可以正确import。此时dipu在父目录下。
export PYTHONPATH=$(pwd)/mmcv:$PYTHONPATH        #将使用diopi编译的mmcv加入pythonpath里

python mmagic/configs/stable_diffusion/stable-diffusion_ddim_denoisingunet_infer.py dipu
python mmagic/configs/stable_diffusion/stable-diffusion_ddim_denoisingunet_compare.py