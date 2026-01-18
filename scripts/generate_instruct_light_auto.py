"""
DistributedDataParallel generation script implemented from scratch.
Using Prot2TextLightDataset instead of Prot2TextInstructDataset.
Generation results will be saved to separate JSON files, and metrics can be further computed with `benchmark.py`.
The script is designed for multi-GPU parallelism on single node. 
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
from datetime import datetime
import json
import os
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

# from dataset import Prot2TextInstructDataset, Prot2TextInstructDataLoader
from dataset.dataset_light_jsonl_dense import Prot2TextLightDataset, Prot2TextLightCollater
from scripts.train_instruct_auto import load_model
import scripts.utils_argparse as utils_argparse


argParser = argparse.ArgumentParser()

argParser.add_argument("--esm_path", type=str)
argParser.add_argument("--llama_path", type=str)
argParser.add_argument("--root_dataset_dir", type=str)
argParser.add_argument("--root_csv_dir", type=str)
argParser.add_argument("--save_generation_dir", type=str)
argParser.add_argument("--save_generation_postfix_identifier", type=str, default=None)
argParser.add_argument("--load_model_checkpoint_path", type=str, default="")
argParser.add_argument("--load_adapter_checkpoint_dir", type=str, default="")

argParser.add_argument("--torch_dtype", type=utils_argparse.str2dtype)
argParser.add_argument("--batch_size_per_device", type=int)
argParser.add_argument("--random_seed", type=int)
argParser.add_argument("--generate_split", type=str)
argParser.add_argument("--debug_trim_generate_split", type=int, default=None)
argParser.add_argument("--max_description_length", type=int, default=1021)  # NEW
argParser.add_argument("--max_sequence_length", type=int, default=512)
argParser.add_argument("--max_generation_length", type=int)
argParser.add_argument("--num_beams", type=int, default=1)
argParser.add_argument("--length_penalty", type=float, default=1.0)
argParser.add_argument("--temperature", type=float, default=1.0)
argParser.add_argument("--do_sample", type=utils_argparse.str2bool, default=False)
argParser.add_argument("--top_p", type=float, default=1.0)
argParser.add_argument("--top_k", type=int, default=50)


def iterative_generation_loop(
        rank: int,
        model: torch.nn.Module,
        data_batch: Dict[str, Any],
        max_generation_length: int,
        num_beams: int,
        length_penalty: float, 
        temperature: float, 
        do_sample: bool, 
        top_p: float,
        top_k: int
) -> torch.Tensor:
    """
    Standard API for different models. Used in `inference_epoch`. 
    1) Prepare inputs for the generation cycle with inference using data_batch from dataloader.
    2) Execute the generation cycle and return the direct output.
    Returned output is a `torch.Tensor` of the generated tokens.
    """
    if isinstance(model, DistributedDataParallel):
        model = model.module  # for wrapper models, get the inner model for generation

    return model.generate(
        inputs=data_batch["input_ids"].to(rank),
        attention_mask=data_batch["attention_mask"].to(rank),
        protein_input_ids=data_batch["protein_input_ids"].to(rank),
        protein_attention_mask=data_batch["protein_attention_mask"].to(rank),
        max_new_tokens=max_generation_length,
        eos_token_id=128009, 
        pad_token_id=128002,
        return_dict_in_generate=False,
        num_beams=num_beams,
        length_penalty=length_penalty,
        temperature=temperature, 
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k
    )


def inference_epoch(
        rank: int, 
        model: DistributedDataParallel,
        # dataloader: Prot2TeqxtInstructDataLoader,
        dataloader: DataLoader, 
        llama_tokenizer: PreTrainedTokenizer,
        args: Dict[str, Any]
): 
    """
    Iterate over all batches for inference with iterative loop. 
    Generation results will be saved to JSON files. 
    """
    model.eval()
    local_names: List[str] = []
    local_predictions: List[str] = []
    local_labels: List[str] = []

    # core loop for batches
    t = tqdm(iter(dataloader))
    for data_batch in t:
        with torch.no_grad():
            output = iterative_generation_loop(
                rank=rank, 
                model=model, 
                data_batch=data_batch, 
                max_generation_length=args["max_generation_length"],
                num_beams=args["num_beams"],
                length_penalty=args["length_penalty"], 
                temperature=args["temperature"], 
                do_sample=args["do_sample"],
                top_p=args["top_p"],
                top_k=args["top_k"]
            )
        local_names.extend(data_batch["name"])
        predicted_texts = llama_tokenizer.batch_decode(output.cpu(), skip_special_tokens=True)
        local_predictions.extend(predicted_texts)
        label_texts = llama_tokenizer.batch_decode(data_batch["description_input_ids"], skip_special_tokens=True)
        local_labels.extend(label_texts)
        t.set_postfix({
            "mode": "inference", 
            "batch_maxlen_gen": output.shape[1], 
            "device": f"rank:{rank}"
        })

    local_json_path = os.path.join(
        args["save_generation_dir"],
        f"generation_{args['save_generation_postfix_identifier']}_rank{rank}.json"
    )
    with open(local_json_path, "w") as file:
        json_dict = {
            name: {"true": label, "pred": prediction}
            for name, label, prediction in zip(local_names, local_labels, local_predictions)
        }
        json.dump(json_dict, file, indent=4)
        print(f"Saving {local_json_path}")


def inference(args: Dict[str, Any]):
    # prepare tokenizers
    esm_tokenizer = AutoTokenizer.from_pretrained(args["esm_path"])
    llama_tokenizer = AutoTokenizer.from_pretrained(
        args["llama_path"],
        pad_token='<|reserved_special_token_0|>'
    )

    # dataset & dataloader
    generate_dataset = Prot2TextLightDataset(
        jsonl_path=os.path.join(args["root_csv_dir"], f"{args['generate_split']}.jsonl")
    )
    generate_collater = Prot2TextLightCollater(
        sequence_tokenizer=esm_tokenizer,
        description_tokenizer=llama_tokenizer,
        mode="inference",
        include_text_fields=True
    )
    generate_loader = DataLoader(
        generate_dataset,
        batch_size=args["batch_size_per_device"],
        shuffle=False,
        collate_fn=generate_collater
    )

    # load model (device_map="auto")
    model = load_model(args)
    model.eval()

    # run inference on single process
    inference_epoch(
        rank=0,
        model=model,
        dataloader=generate_loader,
        llama_tokenizer=llama_tokenizer,
        args=args
    )


def inference_distributed(args: Dict[str, Any]):
    """Core generation process across multiple devices with batches over the whole dataset"""
    torch.multiprocessing.spawn(
        inference_on_device, 
        args=(args["world_size"], args),
        nprocs=args["world_size"],
        join=True
    )


if __name__ == "__main__":
    parsed_args = argParser.parse_args()
    torch.manual_seed(parsed_args.random_seed)
    torch.cuda.manual_seed(parsed_args.random_seed)

    if not os.path.exists(parsed_args.save_generation_dir):
        os.makedirs(parsed_args.save_generation_dir)

    start_timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    parsed_args.save_generation_postfix_identifier = start_timestamp

    inference(parsed_args.__dict__)
