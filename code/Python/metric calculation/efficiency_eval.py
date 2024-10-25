from thop import profile, clever_format


# define model
import argparse
import torch
import yaml
from functools import partial
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

parser = argparse.ArgumentParser()
parser.add_argument('--model_config', type=str,default = 'configs/model_config_imagenet.yaml')
parser.add_argument('--diffusion_config', type=str,default='configs/diffusion_config.yaml')                     
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--save_dir', type=str, default='./VTUAV50_output')
args = parser.parse_args()
model_config = load_yaml(args.model_config)  
model = create_model(**model_config)
model.eval()

# define input
inf_img = torch.randn(1, 1, 512, 512)
x_start = torch.randn((inf_img.repeat(1, 3, 1, 1)).shape)
timesteps = torch.randn(1)

macs, params = profile(model, inputs=(x_start, timesteps))
macs, params = clever_format([macs, params], "%.3f")
print(f"FLOPS: {macs}\nParams: {params}")
