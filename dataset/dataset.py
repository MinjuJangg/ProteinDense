"""
Dataset class for protein function prediction instruction training.

Every sample from the dataset is of following attributes: 
    - graph related features if not ignored:
        - x: (num_nodes, num_node_features)
        - edge_index: (2, num_edges)
        - edge_type: (num_edges, )
    - input ids: 
        - sequence_input_ids: (1, sequence_length+2)  # bos and eos tokens
        - prompt_input_ids: (1, prompt_length)  # with generation prompt head
        - description_input_ids: (1, description_length+1)  # eos token only
    - other attributes:
        - name: str

* No attention mask will be produced at this stage and a dynamic padding will be 
applied in the collate function of the dataloader. 

A chat template is applied in such process to assemble full name and taxon of 
every protein and leave space for sequence embeddings with placeholders. 
The assembled prompt will be of following structure:
    (
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n
        You are a scientific assistant specialized in ...
        <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n
        Name: <NAME> ; Taxon: <TAXON> ; Sequence embeddings: <SEQ>...<SEQ>
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
    )
And the description of the protein will be of following structure:
    (
        Involved in ... Required for ... <|eot_id|>
    )
The template is designed for Meta-Llama-3.1-8B-Instruct-hf and can be adapted 
to other models by calling `process_text` function with the new tokenizer. 

Example of usage:
>>> from transformers import AutoTokenizer
>>> from dataset import Prot2TextInstructDataset
>>> esm_tokenizer = AutoTokenizer.from_pretrained("/data/esm2_t33_650M_UR50D")
>>> llama_tokenizer = AutoTokenizer.from_pretrained(
        "/data/Meta-Llama-3.1-8B-Instruct-hf", 
        pad_token='<|reserved_special_token_0|>'
    )
>>> train_dataset = Prot2TextInstructDataset(
        root_dir="/data/Prot2Text-Llama3-Data/train", 
        csv_path="./data/train.csv", 
        sequence_tokenizer=esm_tokenizer, 
        description_tokenizer=llama_tokenizer, 
        skip_download=False,  # download PDB files
        skip_reload=False,  # construct graph and tokenize text
    )
"""

import multiprocessing as mp
import os
import sys
from typing import Any, Dict, List, Optional, Union

# from graphein.protein.config import ProteinGraphConfig
import pandas as pd
import torch
import torch.utils.data
import torch_geometric
import torch_geometric.data
from transformers import PreTrainedTokenizer
from tqdm import tqdm
import wget

from .pdb2nx import construct_nx_graph
from .nx2pyg import convert_nx_to_pyg
from .utils_dataset import default_graph_process_config


class Prot2TextInstructDataset(torch_geometric.data.Dataset):
    """
    Dataset class for proteins. 
    
    (1) Download PDB files from AlphaFoldDB if not skipped, then
    (2) preprocess graph and textual features if not skipped to prepare for the 
    formation of :class:`Prot2TextDataLoader`. 
    
    * Use multiple :class:`Prot2TextDataset` instead of one for multiple 
    splits of the dataset. 
    * root_dir should be the root directory of each split.

    Args:
        root_dir:
            Root directory of the dataset where root_dir/raw and 
            root_dir/processed will be prepared. If the dataset is 
            of a particular split of the whole dataset, the directory 
            of the split should be passed instead.
        csv_path:
            Path of the csv file containing the uniprot ids, amino-acid 
            sequence and the description.
        sequence_tokenizer:
            Tokenizer for protein amino-acid sequences. Tokenization 
            will be done in preprocessing without pad token, and padding 
            will be added in dataloader when forming data batch. The 
            tokenizer should have :attr:`pad_token` and :attr:`pad_token_id` 
            attributes which will be used in collate function of the 
            dataloader for dynamic padding.
        description_tokenizer:
            Tokenizer for protein functionality descriptions. The 
            tokenizer should add bos token but not add eos token by 
            itself for the consistency with prompt tokenization. However, 
            the tokenizer should have :attr:`eos_token` and 
            :attr:`eos_token_id` attributes which will be used in 
            preprocessing to add eos token at the end of the description 
            and the label. The tokenizer should also have :attr:`pad_token` 
            and :attr:`pad_token_id` attributes which will be used in 
            collate function of the dataloader for dynamic padding.
        alphafold_base_url:
            Base download URL of AlphaFoldDB to download PDB files from. 
            The full download link will be :url:`{alphafold_base_url}/AF-
            {uniprot_id}-F1-model_v{alphafold_version}.pdb`.
        alphafold_version:
            Version of the AlphaFoldDB to download PDB files from.
        graph_process_config:
            Configuration specifying features of nodes, edges and the 
            whole graph to be used for graph construction from PDB files. 
            If not passed, Default configuration will be used.
        skip_download:
            Force preprocessing to skip downloading procedure and to use 
            existing PDB files only. Otherwise, downloading procedure 
            will be skipped only if every uniprot id from CSV file has 
            corresponding PDB file under `root_dir/raw`.
        skip_reload:
            Force preprocessing to skip formation of graphs from PDB files 
            and to use existing PyG tensor files only. Otherwise, processing 
            procedure will be skipped only if every PDB file under 
            `root_dir/raw` has corresponding PyG tensor file under 
            `root_dir/processed`.
        num_processes:
            Number of parallel processes in formation of the graph. If 
            not specified, all logical CPU threads will be used by default.
        ignore_graph_features:
            Ignore graph related features while loading data. This does 
            not affect the preprocessing behavior and graph related features 
            will be processed then saved.
        max_sequence_length:
            Maximum length of protein amino-acid sequence. Samples with 
            longer sequences will be trimmed if such value passed. Such 
            behavior will lose part of the information but can avoid 
            Out-Of-Memory error in training. This does not affect the 
            preprocessing behavior and whole sequences will be processed 
            then saved.
        max_description_length:
            Maximum length of protein description. Samples with longer 
            descriptions will be trimmed if such value passed. Such 
            behavior will lose part of the information but can avoid 
            Out-Of-Memory error in training. This does not affect the 
            preprocessing behavior and whole descriptions will be processed 
            then saved.
        system_message:
            The system message to be used in the chat template. 
        placeholder_token:
            The placeholder token to be used in the chat template that will be 
            replaced by the output of encoder as sequence embeddings.
        kwargs:
            Additional arguments controlling the behavior of the PyG 
            dataset inherited from the base dataset class. Read the 
            documentation from :class:`torch_geometric.data.Dataset` 
            for more information.
    """
    def __init__(
            self,
            root_dir: Union[str, os.PathLike],
            csv_path: Union[str, os.PathLike],
            sequence_tokenizer: PreTrainedTokenizer,
            description_tokenizer: PreTrainedTokenizer,
            alphafold_base_url: str = "https://alphafold.ebi.ac.uk/files/",
            alphafold_version: int = 4,
            graph_process_config: ProteinGraphConfig = default_graph_process_config,
            skip_download: bool = False,
            skip_reload: bool = False,
            num_processes: Optional[int] = None,
            ignore_graph_features: bool = True,
            max_sequence_length: Optional[int] = 1021,
            max_description_length: Optional[int] = 512,
            system_message: str = (
                "You are a scientific assistant specialized in protein function "
                "predictions. Given the sequence embeddings and other information "
                "of a protein, describe its function clearly and concisely in "
                "professional language. "
            ), 
            placeholder_token: str = '<|reserved_special_token_1|>', 
            **kwargs,
    ):
        self.root_dir = root_dir
        self.uniprot_df = pd.read_csv(csv_path)
        self.sequence_tokenizer = sequence_tokenizer
        self.description_tokenizer = description_tokenizer
        self.alphafold_base_url = alphafold_base_url
        self.alphafold_version = alphafold_version
        self.graph_process_config = graph_process_config
        self.skip_download = skip_download
        self.skip_reload = skip_reload
        self.num_processes = num_processes
        self.ignore_graph_features = ignore_graph_features
        self.max_sequence_length = max_sequence_length
        self.max_description_length = max_description_length
        self.system_message = system_message
        self.placeholder_token = placeholder_token

        self.usable_file_names: List[Union[str, os.PathLike]] = []
        super().__init__(root=root_dir, **kwargs)  # first download then process
        self.update_usable_file_names()

    def download(self, overwrite_existing: bool = False):
        """
        Downloads PDB files from AlphaFoldDB to :attr:`self.raw_dir` folder. If 
        such file already exists, the download will be skipped. Unsuccessful 
        attempt of downloading will not crash the script but exceptions will be 
        noted.
        """
        if self.skip_download:
            return
        assert self.alphafold_base_url is not None, (
            "Downloading requested but base URL of AlphaFoldDB is not set. "
        )
        assert self.alphafold_version is not None, (
            "Downloading requested but version of AlphaFoldDB is not set. "
        )
        for raw_file_name in tqdm(self.raw_file_names):
            raw_file_name = raw_file_name
            raw_file_path = os.path.join(self.raw_dir, raw_file_name)
            full_url = self.alphafold_base_url + raw_file_name
            if overwrite_existing or not os.path.exists(raw_file_path):
                try:
                    wget.download(full_url, out=raw_file_path)
                except Exception as exception:
                    if self.log:
                        print(
                            f"Download of {raw_file_name} failed due to exception: "
                            f"{exception.__class__.__name__}: {exception}",
                            file=sys.stderr
                        )

    def process(self, overwrite_existing: bool = False):
        """
        Preprocesses and converts PDB files to pytorch tensor files. Parallel 
        multiprocessing approach applied for graph construction. Unsuccessful 
        attempt of graph processing will not crash the script but exceptions will 
        be noted. Textual features will be encoded and added to tensor files in 
        the end without parallelism.
        """
        if self.skip_reload:
            return
        assert self.sequence_tokenizer is not None, (
            "Processing requested but sequence tokenizer is not set."
        )
        assert self.description_tokenizer is not None, (
            "Processing requested but description tokenizer is not set."
        )

        # graph construction: convert PDB files to tensor files
        with mp.Pool(processes=self.num_processes) as pool:
            for raw_file_name in self.raw_file_names:
                raw_file_path = os.path.join(self.raw_dir, raw_file_name)
                processed_file_path = os.path.join(
                    self.processed_dir, 
                    os.path.splitext(raw_file_name)[0] + ".pt"
                )
                if overwrite_existing or not os.path.exists(processed_file_path):
                    pool.apply_async(
                        self.process_graph, 
                        args=(raw_file_path, processed_file_path)
                    )
            pool.close()
            pool.join()
        self.update_usable_file_names()

        # textual feature tokenization: add tokenized features to tensor files
        self.process_text()
        self.update_usable_file_names()
        
    def process_graph(
            self,
            raw_file_path: Union[str, os.PathLike],
            processed_file_path: Union[str, os.PathLike]
    ):
        """
        Preprocesses and converts single PDB file to pytorch tensor file. Unit 
        function for multiprocessing.
        """
        uniprot_id = os.path.split(raw_file_path)[1].split("-")[1]
        try:
            nx_graph = construct_nx_graph(
                config=self.graph_process_config, 
                pdb_path=raw_file_path
            )
            data = convert_nx_to_pyg(nx_graph)
            torch.save(data, processed_file_path)
        except Exception as exception:
            if self.log:
                print(
                    f"Graph processing of {uniprot_id} failed due to exception: "
                    f"{exception.__class__.__name__}: {exception}",
                    file=sys.stderr
                )

    def process_text(
            self,
            new_sequence_tokenizer: Optional[PreTrainedTokenizer] = None,
            new_description_tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        """
        Apply tokenizers to add tokenized protein amino-acid sequences and 
        protein functionality descriptions to processed pytorch tensor files. 
        Such process will not download and process graphs again, but only 
        add/modify the textual features in processed Data objects. Can be used 
        to change tokenizers on existing processed dataset.
        """
        if new_sequence_tokenizer is not None:
            self.sequence_tokenizer = new_sequence_tokenizer
        if new_description_tokenizer is not None:
            self.description_tokenizer = new_description_tokenizer
        for processed_file_name in tqdm(self.usable_file_names):
            processed_file_path = os.path.join(self.processed_dir, processed_file_name)
            try:
                uniprot_id = os.path.split(processed_file_name)[1].split("-")[1]
                data = torch.load(processed_file_path)
                data.update(self._compose_and_tokenize_chat(uniprot_id))
                torch.save(data, processed_file_path)
            except Exception as exception:
                if self.log:
                    print(
                        f"Text processing of {processed_file_name} failed due to exception: "
                        f"{exception.__class__.__name__}: {exception}",
                        file=sys.stderr
                    )

    def _compose_and_tokenize_chat(self, uniprot_id: str) -> Dict[str, torch.Tensor]:
        """
        (1) Matches corresponding row in CSV DataFrame of the given uniprot id, 
        (2) trim sequence and description to avoid OOM error,
        (3) apply chat template to form the prompt with placeholder, and 
        (4) tokenizes amino-acid sequence, prompt and description. 
        
        The eos token will be added at the end of the description and of the 
        label before tokenization.
        """
        # (1) match row and extract features
        filtered_df = self.uniprot_df.loc[
            self.uniprot_df['AlphaFoldDB'] == uniprot_id
        ]
        sequence = filtered_df["sequence"].values[0]
        description = filtered_df["function"].values[0]
        fullname = filtered_df["Full Name"].values[0]
        taxon = filtered_df["taxon"].values[0]
        fullname = "unknown" if pd.isna(fullname) else fullname
        taxon = "unknown" if pd.isna(taxon) else taxon

        # (2) trim sequence and description
        if self.max_description_length is not None:
            description_ids = self.description_tokenizer(
                [description], 
                add_special_tokens=False, 
                return_tensors="pt"
            )["input_ids"]
            if description_ids.size(-1) > self.max_description_length:
                description_ids = description_ids[:, :self.max_description_length]
                description = self.description_tokenizer.decode(description_ids[0])
        if self.max_sequence_length is not None:
            if len(sequence) > self.max_sequence_length:
                sequence = sequence[:self.max_sequence_length]

        # (3) apply chat template then tokenize the prompt
        user_message = (
            "Protein name: " + fullname 
            + " ; Taxon: " + taxon 
            + " ; Sequence embeddings: " 
            + self.placeholder_token * (len(sequence) + 2)  # bos and eos tokens
        )
        prompt_conversation = [
            {"role": "system", "content": self.system_message}, 
            {"role": "user", "content": user_message}
        ]
        prompt_ids = self.description_tokenizer.apply_chat_template(
            prompt_conversation,
            add_generation_prompt=True, 
            tokenize=True, 
            padding=False, 
            return_tensors="pt"
        )

        # (4) tokenize sequence and description
        sequence_ids = self.sequence_tokenizer(
            [sequence], 
            add_special_tokens=True, 
            return_attention_mask=False, 
            return_tensors="pt"
        )["input_ids"]
        description_ids = self.description_tokenizer(
            [description + self.description_tokenizer.eos_token],
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors="pt"
        )["input_ids"]

        # tensors should be kept of shape (1, seq_length) for formation of batch 
        # in dataloader
        return {
            "sequence_input_ids": sequence_ids,
            "prompt_input_ids": prompt_ids,
            "description_input_ids": description_ids,
        }

    @property
    def raw_file_names(self) -> List[str]:
        """
        The name of files in `self.raw_dir` folder that must be present in order 
        to skip downloading. Required in the parent class. 
        """
        uniprot_ids = set(self.uniprot_df.AlphaFoldDB)
        return [
            f"AF-{uniprot_id}-F1-model_v{self.alphafold_version}.pdb" 
            for uniprot_id in uniprot_ids
        ]

    @property
    def processed_file_names(self) -> List[str]:
        """
        The name of files in `self.processed_dir` folder that must be present in 
        order to skip processing. Required in the parent class. 
        """
        return [
            os.path.splitext(raw_file_name)[0]+".pt" 
            for raw_file_name in os.listdir(self.raw_dir)
        ]

    def update_usable_file_names(self):
        """
        Updates stored name of files in `self.processed_dir` folder that are 
        ready to be used for the dataset. Usable file names shall be updated in 
        initialization and preprocessing but not recommended in actual usage to
        maintain consistency of the order of file names.
        """
        existing_file_names = os.listdir(self.processed_dir)
        for special_file_name in ['pre_transform.pt', 'pre_filter.pt']:
            if special_file_name in existing_file_names:
                existing_file_names.remove(special_file_name)
        self.usable_file_names = sorted(existing_file_names)

    def len(self) -> int:
        """Required in the parent class."""
        return len(self.usable_file_names)

    def __len__(self) -> int:
        return self.len()

    def get(self, idx: int, debug_mode: bool = False) -> Any:
        data = torch.load(os.path.join(self.processed_dir, self.usable_file_names[idx]))
        if debug_mode: # return all items stored in the file
            return data
        if self.ignore_graph_features:
            return torch_geometric.data.Data(
                sequence_input_ids=data['sequence_input_ids'],
                prompt_input_ids=data["prompt_input_ids"],
                description_input_ids=data['description_input_ids'],
                name=data["name"], 
            )
        else: 
            return torch_geometric.data.Data(
                x=data['x'],
                edge_index=data['edge_index'],
                edge_type=data['edge_type'],
                sequence_input_ids=data['sequence_input_ids'],
                prompt_input_ids=data["prompt_input_ids"],
                description_input_ids=data['description_input_ids'],
                name=data["name"],
            )
