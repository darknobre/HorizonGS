# mainly used for generating config file for per-chunk training
import copy
import sys
import textwrap
import numpy as np
import torch
import yaml
import os
import open3d as o3d
import argparse
import copy
from types import SimpleNamespace
from decimal import Decimal
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from preprocess.generate_chunks_config import generate_chunks_config
from utils.general_utils import *

def format_params(param_dict):
    formatted_str = "{\n"  # Start with an opening brace and a newline
    for key, value in param_dict.items():
        if isinstance(value, float):
            value = format(value, '.6f')
        if value == '':
            formatted_str += f"    {key}: '{value}',\n"
        else:
            formatted_str += f"    {key}: {value},\n"  # Add each key-value pair with indentation
    formatted_str = formatted_str.rstrip(',\n') + "\n"  # Remove the last comma and newline
    formatted_str += "}"  # Close the brace
    return formatted_str  # Return the formatted string

def output_yaml(lp, pp, op, save_path):
    yaml_output = f"""
# Model Parameters
model_params: {format_params(vars(lp))}

# Pipeline Parameters
pipeline_params: {format_params(vars(pp))}
    
# Optimization Parameters
optim_params: {format_params(vars(op))}
    """

    yaml_output = textwrap.dedent(yaml_output)

    with open(save_path, 'w') as file:
        file.write(yaml_output)


def generate_config(dp, lp):
    """
    Generate the config file for each chunk
    """
    config_path = os.path.dirname(os.path.dirname(dp.config_path))
    config_name = os.path.basename(dp.config_path)

    lp_c = copy.deepcopy(lp)
    lp_c.source_path = dp.source_path
    lp_c.scene_name = lp.scene_name+"/coarse"
    
    with open(os.path.join(dp.pp_opt_yaml_path, "coarse.yaml")) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        pp_c, op_c = parse_cfg_pp_op(cfg)

    lp_f = copy.deepcopy(lp)
    lp_f.source_path = dp.source_path
    lp_f.scene_name = lp.scene_name+"/fine"
    lp_f.pretrained_checkpoint = os.path.join("outputs", lp.dataset_name, lp.scene_name, "coarse")
    if lp.model_config["kwargs"]["appearance_dim"] > 0:
        lp_f.global_appearance = os.path.join("outputs", lp.dataset_name, lp.scene_name, "coarse")

    with open(os.path.join(dp.pp_opt_yaml_path, "fine.yaml")) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        pp_f, op_f = parse_cfg_pp_op(cfg)
    
    output_yaml(lp_c, pp_c, op_c, os.path.join(os.path.join(os.path.dirname(dp.config_path), "coarse.yaml")))
    output_yaml(lp_f, pp_f, op_f, os.path.join(os.path.join(os.path.dirname(dp.config_path), "fine.yaml")))

if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(
        description="cluster poses and pcd")
        
        parser.add_argument('--config', type=str, help='partition config file path')

        args = parser.parse_args()
        return args
    args = parse_args()
    
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        dp, lp = parse_cfg_dp(cfg)
    
    dp.config_path = args.config
    generate_chunks_config(dp, lp)