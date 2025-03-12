#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    masks = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        render_image = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
        render_mask = tf.to_tensor(render).unsqueeze(0)[:, 3:4, :, :].cuda()
        render_image = render_image * render_mask
        gt_image = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
        gt_mask = tf.to_tensor(gt).unsqueeze(0)[:, 3:4, :, :].cuda()
        gt_image = gt_image * gt_mask
        renders.append(render_image)
        gts.append(gt_image)
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / "train"

        for method in os.listdir(test_dir):
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            base_method_dir = test_dir / method
            method_dir = base_method_dir / "aerial"
            json_path = method_dir / "per_view_count.json" 
            if os.path.exists(method_dir) and os.path.exists(json_path):
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                json_path = method_dir / "per_view_count.json" 
                renders, gts, image_names = readImages(renders_dir, gt_dir)
                
                json_file = open(json_path)
                gs_data = json.load(json_file)
                json_file.close()
                ssims = []
                psnrs = []
                lpipss = []
                gss = []
                
                for idx, image_name in tqdm(enumerate(image_names), desc="Metric evaluation progress"):
                    gss.append(gs_data[image_name])
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

                print("  AERIAL_PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  AERIAL_SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  AERIAL_LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("  AERIAL_GS_NUMS: {:>12.7f}".format(torch.tensor(gss).float().mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({
                    "AERIAL_PSNR": torch.tensor(psnrs).mean().item(),
                    "AERIAL_SSIM": torch.tensor(ssims).mean().item(),
                    "AERIAL_LPIPS": torch.tensor(lpipss).mean().item(),
                    "AERIAL_GS_NUMS": torch.tensor(gss).float().mean().item()
                    })

            method_dir = base_method_dir / "street"
            json_path = method_dir / "per_view_count.json" 
            if os.path.exists(method_dir) and os.path.exists(json_path):
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)
                
                json_file = open(json_path)
                gs_data = json.load(json_file)
                json_file.close()
                ssims = []
                psnrs = []
                lpipss = []
                gss = []
                
                for idx, image_name in tqdm(enumerate(image_names), desc="Metric evaluation progress"):
                    gss.append(gs_data[image_name])
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

                print("  STREET_PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  STREET_SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  STREET_LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("  STREET_GS_NUMS: {:>12.7f}".format(torch.tensor(gss).float().mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({
                    "STREET_PSNR": torch.tensor(psnrs).mean().item(),
                    "STREET_SSIM": torch.tensor(ssims).mean().item(),
                    "STREET_LPIPS": torch.tensor(lpipss).mean().item(),
                    "STREET_GS_NUMS": torch.tensor(gss).float().mean().item()
                    })

        with open(scene_dir + "/results.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)

if __name__ == "__main__":
    lpips_fn = lpips.LPIPS(net='vgg').cuda()
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
