##! consolidate/merge chunks

# 1. load all centers
# 2. delete gs primitives if it's closer than another center

from argparse import ArgumentParser
import yaml
import os
from os import makedirs
import numpy as np
from plyfile import PlyData, PlyElement
from types import SimpleNamespace
import subprocess

from metrics import readImages
from scene import Scene
from utils.system_utils import searchForMaxIteration
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

import torch
import torchvision
import json
import time
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
from random import randint
import sys
from utils.general_utils import parse_cfg_dp, safe_state, parse_cfg
from tqdm import tqdm
from utils.image_utils import psnr, save_rgba
from utils.loss_utils import l1_loss, ssim
import lpips
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

def construct_list_of_attributes_explict(l, max_sh_degree):
    # All channels except the 3 DC
    for i in range(3):
        l.append('f_dc_{}'.format(i))
    for i in range(3*(max_sh_degree + 1) ** 2 - 3):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(3):
        l.append('scale_{}'.format(i))
    for i in range(4):
        l.append('rot_{}'.format(i))
    return l    
    
def consolidate(dataset, model_path):
    # 1. load all attr
    # 2. delete gs primitives if it's closer than another center
    scale = dataset.scale
    plane_index = [index for index, value in enumerate(dataset.xyz_plane) if value == 1]
    assert len(plane_index) == 2
    
    partitions = load_partition_data(os.path.join(dataset.source_path, "chunks",
                                     f"init_ply_coverage_{dataset.n_width*dataset.n_height}parts_{dataset.visible_rate}.th")) #! notice the path folder
    merge_path = os.path.join(model_path, "merged_model")
    os.makedirs(merge_path, exist_ok=True)
    chunks_path = os.path.join(model_path, "chunks")
    xyz, feature_dc, feature_rest, opacity, scaling, rots = [], [], [], [], [], []
    for partition_id, partition in partitions.items():
        unique_id = sorted(os.listdir(os.path.join(chunks_path, partition_id)))[-1]
        print(partition_id)
        per_chunk_path = os.path.join(chunks_path, partition_id, unique_id)
        with open(os.path.join(per_chunk_path, "config.yaml")) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            lp, op, pp = parse_cfg(cfg)
            lp.model_path = args.model_path
        
        point_cloud_path = os.path.join(per_chunk_path, "point_cloud")
        loaded_iter = searchForMaxIteration(os.path.join(point_cloud_path))
        ckpt_path = os.path.join(point_cloud_path, f"iteration_{loaded_iter}", "point_cloud_explicit.ply")
        
        modules = __import__('scene')
        model_config = lp.model_config
        gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
        gaussians.eval()
        gaussians.load_explicit(ckpt_path)

        # filter by origin bounding box
        x_bounds, y_bounds = partition["true_bounds"]
        
        print(f"{partition_id}: origin gaussians num {gaussians._xyz.shape[0]}")
        mask = (
            (gaussians._xyz[:, plane_index[0]] >= x_bounds[0]/scale)
            & (gaussians._xyz[:, plane_index[0]] <= x_bounds[1]/scale)
            & (gaussians._xyz[:, plane_index[1]] >= y_bounds[0]/scale)
            & (gaussians._xyz[:, plane_index[1]] <= y_bounds[1]/scale)
        )
        
        gaussians._xyz = gaussians._xyz[mask]
        gaussians._features_dc = gaussians._features_dc[mask]
        gaussians._features_rest = gaussians._features_rest[mask]
        gaussians._opacity = gaussians._opacity[mask]
        gaussians._scaling = gaussians._scaling[mask]
        gaussians._rotation = gaussians._rotation[mask]

        print(f"{partition_id}: filtered gaussians num {gaussians._xyz.shape[0]}")

        # merge
        xyz.append(gaussians._xyz.detach().cpu().numpy())
        feature_dc.append(gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy())
        feature_rest.append(gaussians._features_rest.transpose(1, 2).flatten(start_dim=1).contiguous().detach().cpu().numpy())
        opacity.append(gaussians._opacity.detach().cpu().numpy())
        scaling.append(gaussians._scaling.detach().cpu().numpy())
        rots.append(gaussians._rotation.detach().cpu().numpy())
    
    xyz= np.concatenate(xyz)
    feature_dc = np.concatenate(feature_dc)
    feature_rest = np.concatenate(feature_rest)
    opacity = np.concatenate(opacity)
    scaling = np.concatenate(scaling)
    rots = np.concatenate(rots)

    # save
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes_explict(['x', 'y', 'z'], gaussians.max_sh_degree)]
    
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, feature_dc, feature_rest, opacity, scaling, rots), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    os.makedirs(os.path.join(os.path.join(merge_path, "point_cloud", f"iteration_{loaded_iter}")), exist_ok=True)
    PlyData([el]).write(os.path.join(merge_path, "point_cloud", f"iteration_{loaded_iter}", "point_cloud_explicit.ply"))

def consolidate_lod(dataset, model_path):
    # 1. load all attr
    # 2. delete gs primitives if it's closer than another center
    scale = dataset.scale
    plane_index = [index for index, value in enumerate(dataset.xyz_plane) if value == 1]
    assert len(plane_index) == 2
    
    partitions = load_partition_data(os.path.join(dataset.source_path, "chunks",
                                     f"init_ply_coverage_{dataset.n_width*dataset.n_height}parts_{dataset.visible_rate}.th")) #! notice the path folder
    merge_path = os.path.join(model_path, "merged_model")
    os.makedirs(merge_path, exist_ok=True)
    chunks_path = os.path.join(model_path, "chunks")
    xyz, level, extra_level, feature_dc, feature_rest, opacity, scaling, rots = [], [], [], [], [], [], [], []
    for partition_id, partition in partitions.items():
        unique_id = os.listdir(os.path.join(chunks_path, partition_id))[0]
        per_chunk_path = os.path.join(chunks_path, partition_id, unique_id)
        with open(os.path.join(per_chunk_path, "config.yaml")) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            lp, op, pp = parse_cfg(cfg)
            lp.model_path = args.model_path
        
        point_cloud_path = os.path.join(per_chunk_path, "point_cloud")
        loaded_iter = searchForMaxIteration(os.path.join(point_cloud_path))
        ckpt_path = os.path.join(point_cloud_path, f"iteration_{loaded_iter}", "point_cloud_explicit.ply")
        
        modules = __import__('scene')
        model_config = lp.model_config
        gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
        gaussians.eval()
        gaussians.load_explicit(ckpt_path)

        # filter by origin bounding box
        x_bounds, y_bounds = partition["true_bounds"]
        
        print(f"{partition_id}: origin gaussians num {gaussians._xyz.shape[0]}")
        mask = (
            (gaussians._xyz[:, plane_index[0]] >= x_bounds[0]/scale)
            & (gaussians._xyz[:, plane_index[0]] <= x_bounds[1]/scale)
            & (gaussians._xyz[:, plane_index[1]] >= y_bounds[0]/scale)
            & (gaussians._xyz[:, plane_index[1]] <= y_bounds[1]/scale)
        )
        
        gaussians._xyz = gaussians._xyz[mask]
        gaussians._level = gaussians._level[mask]
        gaussians._extra_level = gaussians._extra_level[mask]
        gaussians._features_dc = gaussians._features_dc[mask]
        gaussians._features_rest = gaussians._features_rest[mask]
        gaussians._opacity = gaussians._opacity[mask]
        gaussians._scaling = gaussians._scaling[mask]
        gaussians._rotation = gaussians._rotation[mask]

        print(f"{partition_id}: filtered gaussians num {gaussians._xyz.shape[0]}")

        # merge
        xyz.append(gaussians._xyz.detach().cpu().numpy())
        level.append(gaussians._level.detach().cpu().numpy())
        extra_level.append(gaussians._extra_level.detach().cpu().numpy())
        feature_dc.append(gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy())
        feature_rest.append(gaussians._features_rest.transpose(1, 2).flatten(start_dim=1).contiguous().detach().cpu().numpy())
        opacity.append(gaussians._opacity.detach().cpu().numpy())
        scaling.append(gaussians._scaling.detach().cpu().numpy())
        rots.append(gaussians._rotation.detach().cpu().numpy())
    
    xyz= np.concatenate(xyz)
    level = np.concatenate(level)
    extra_level = np.concatenate(extra_level).reshape(-1, 1)
    feature_dc = np.concatenate(feature_dc)
    feature_rest = np.concatenate(feature_rest)
    opacity = np.concatenate(opacity)
    scaling = np.concatenate(scaling)
    rots = np.concatenate(rots)

    # save
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes_explict(['x', 'y', 'z', 'level', 'extra_level'], gaussians.max_sh_degree)]
    
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, level, extra_level, feature_dc, feature_rest, opacity, scaling, rots), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    os.makedirs(os.path.join(os.path.join(merge_path, "point_cloud", f"iteration_{loaded_iter}")), exist_ok=True)
    plydata = PlyData([el], obj_info=[
    'standard_dist {:.6f}'.format(gaussians.standard_dist),
    'aerial_levels {:.6f}'.format(gaussians.aerial_levels),
    'street_levels {:.6f}'.format(gaussians.street_levels),
    ])
    plydata.write(os.path.join(merge_path, "point_cloud", f"iteration_{loaded_iter}", "point_cloud_explicit.ply"))


def load_partition_data(ckpt_path, train_cameras=None):
    partitions = torch.load(ckpt_path)
    if train_cameras is not None:
        for key in partitions.keys():
            cameras = [train_cameras[idx] for idx in partitions[key]["indices"]]
            partitions[key]["cameras"] = cameras
    return partitions

def render_set(model_path, name, iteration, views, gaussians, background, add_aerial, add_street):
    if add_aerial:
        aerial_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "renders")
        aerial_error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "errors")
        aerial_gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "gt")
        makedirs(aerial_render_path, exist_ok=True)
        makedirs(aerial_error_path, exist_ok=True)
        makedirs(aerial_gts_path, exist_ok=True)
    if add_street:
        street_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "street", "renders")
        street_error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "street", "errors")
        street_gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "street", "gt")
        makedirs(street_render_path, exist_ok=True)
        makedirs(street_error_path, exist_ok=True)
        makedirs(street_gts_path, exist_ok=True)

    modules = __import__('gaussian_renderer')

    street_t_list = []
    street_visible_count_list = []
    street_per_view_dict = {}
    street_views = [view for view in views if view.image_type=="street"]
    pipe = SimpleNamespace(add_prefilter=False)
    for idx, view in enumerate(tqdm(street_views, desc="Street rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()        
        render_pkg = getattr(modules, 'render')(view, gaussians, pipe, background)
        torch.cuda.synchronize();t_end = time.time()

        street_t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()

        # gts
        gt = view.original_image.cuda()
        alpha_mask = view.alpha_mask.cuda()
        rendering = torch.cat([rendering, alpha_mask], dim=0)
        gt = torch.cat([gt, alpha_mask], dim=0)
        
        # error maps
        if gt.device != rendering.device:
            rendering = rendering.to(gt.device)
        errormap = (rendering - gt).abs()

        save_rgba(rendering, os.path.join(street_render_path, '{0:05d}'.format(idx) + ".png"))
        save_rgba(errormap, os.path.join(street_error_path, '{0:05d}'.format(idx) + ".png"))
        save_rgba(gt, os.path.join(street_gts_path, '{0:05d}'.format(idx) + ".png"))
        street_visible_count_list.append(visible_count)
        street_per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
    
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "street", "per_view_count.json"), 'w') as fp:
        json.dump(street_per_view_dict, fp, indent=True)
    
    aerial_t_list = []
    aerial_visible_count_list = []
    aerial_per_view_dict = {}
    aerial_views = [view for view in views if view.image_type=="aerial"]
    for idx, view in enumerate(tqdm(aerial_views, desc="Aerial rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        render_pkg = getattr(modules, 'render')(view, gaussians, pipe, background)
        torch.cuda.synchronize();t_end = time.time()

        aerial_t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()

        # gts
        gt = view.original_image.cuda()
        alpha_mask = view.alpha_mask.cuda()
        rendering = torch.cat([rendering, alpha_mask], dim=0)
        gt = torch.cat([gt, alpha_mask], dim=0)
        
        # error maps
        if gt.device != rendering.device:
            rendering = rendering.to(gt.device)
        errormap = (rendering - gt).abs()

        save_rgba(rendering, os.path.join(aerial_render_path, '{0:05d}'.format(idx) + ".png"))
        save_rgba(errormap, os.path.join(aerial_error_path, '{0:05d}'.format(idx) + ".png"))
        save_rgba(gt, os.path.join(aerial_gts_path, '{0:05d}'.format(idx) + ".png"))
        aerial_visible_count_list.append(visible_count)
        aerial_per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "per_view_count.json"), 'w') as fp:
        json.dump(aerial_per_view_dict, fp, indent=True)
    
    return aerial_visible_count_list, street_visible_count_list

def render_sets(dataset, iteration, skip_train=False, skip_test=False, explicit=True):
    with torch.no_grad():
        modules = __import__('scene')
        model_config = dataset.model_config
        gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, explicit=explicit)
        gaussians.eval()

        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            aerial_visible_count, street_visible_count = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, scene.background, dataset.add_aerial, dataset.add_street)
        if not skip_test:
            aerial_visible_count, street_visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, scene.background, dataset.add_aerial, dataset.add_street)
    
    return aerial_visible_count, street_visible_count

def evaluate(model_paths, eval_name, aerial_visible_count=None, street_visible_count=None):
    
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / eval_name

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        base_method_dir = test_dir / method
        method_dir = base_method_dir / "aerial" 
        if os.path.exists(method_dir):
            gt_dir = method_dir/ "gt"
            renders_dir = method_dir / "renders"
            renders, gts, image_names = readImages(renders_dir, gt_dir)

            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

            print(f"model_paths: \033[1;35m{model_paths}\033[0m")
            print("  AERIAL_PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
            print("  AERIAL_SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
            print("  AERIAL_LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
            print("  AERIAL_GS_NUMS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(aerial_visible_count).float().mean(), ".5"))
            print("")
            
            full_dict[scene_dir][method].update({
                "AERIAL_PSNR": torch.tensor(psnrs).mean().item(),
                "AERIAL_SSIM": torch.tensor(ssims).mean().item(),
                "AERIAL_LPIPS": torch.tensor(lpipss).mean().item(),
                "AERIAL_GS_NUMS": torch.tensor(aerial_visible_count).float().mean().item(),
                })

            per_view_dict[scene_dir][method].update({
                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                "GS_NUMS": {name: vc for vc, name in zip(torch.tensor(aerial_visible_count).tolist(), image_names)}
                })

        method_dir = base_method_dir / "street" 
        if os.path.exists(method_dir):
            gt_dir = method_dir/ "gt"
            renders_dir = method_dir / "renders"
            renders, gts, image_names = readImages(renders_dir, gt_dir)

            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

            print(f"model_paths: \033[1;35m{model_paths}\033[0m")
            print("  STREET_PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
            print("  STREET_SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
            print("  STREET_LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
            print("  STREET_GS_NUMS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(street_visible_count).float().mean(), ".5"))
            print("")
            
            full_dict[scene_dir][method].update({
                "STREET_PSNR": torch.tensor(psnrs).mean().item(),
                "STREET_SSIM": torch.tensor(ssims).mean().item(),
                "STREET_LPIPS": torch.tensor(lpipss).mean().item(),
                "STREET_GS_NUMS": torch.tensor(street_visible_count).float().mean().item(),
                })

            per_view_dict[scene_dir][method].update({
                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                "GS_NUMS": {name: vc for vc, name in zip(torch.tensor(street_visible_count).tolist(), image_names)}
                })


    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Merge script parameters")
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument('--config', type=str, required=True, help='partition config file path')
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        dp = parse_cfg_dp(cfg)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    if "LoD" in dp.model_config["name"]:
        consolidate_lod(dp, model_path=args.model_path)
    else:    
        consolidate(dp, model_path=args.model_path)
    print("Merge completed")
    
    dp.model_path = os.path.join(args.model_path, "merged_model")
    aerial_visible_count, street_visible_count = render_sets(dp, -1, skip_train=True, skip_test=False)
    print("Render completed")
    
    eval_name = 'test' if dp.eval else 'train'
    evaluate(dp.model_path, eval_name, aerial_visible_count=aerial_visible_count, street_visible_count=street_visible_count)
    print("Evaluating completed.")