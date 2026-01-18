"""Utilities for argparse configuration."""

import torch


def str2bool(string: str = "") -> bool:
    """
    Override default build-in bool() function to facilitate argparse configuration.

    Example of usage:
        >>> import argparse
        >>> from utils import str2bool
        >>> argParser = argparse.ArgumentParser()
        >>> argParser.add_argument(
                "--debug_model", 
                default=False, 
                type=str2bool, 
                help="boolean flag to debug model"
            )
    """
    negative_keywords = ["false", "no", "none", "negative", "off", "disable", "f", "0"]
    if not string or any([string.lower() == keyword for keyword in negative_keywords]):
        return False
    return True


def str2dtype(string: str = "") -> torch.dtype: 
    """
    Convert string to corresponding torch datatype to facilitate argparse 
    configuration for autocast dtype. 
    """
    if not string: 
        return torch.float32
    string = string.lower()
    bf16_keywords = ["torch.bfloat16", "bfloat16", "bf16"]
    fp16_keywords = ["torch.float16", "float16", "fp16", "16", "half"]
    int8_keywords = ["torch.int8", "int8", "8"]
    int4_keywords = ["torch.int4", "int4", "4"]
    if any([string == keyword for keyword in bf16_keywords]): 
        return torch.bfloat16
    elif any([string == keyword for keyword in fp16_keywords]): 
        return torch.float16
    elif any([string == keyword for keyword in int8_keywords]): 
        return torch.int8
    elif any([string == keyword for keyword in int4_keywords]): 
        return torch.int4
    else: 
        return torch.float32    
    