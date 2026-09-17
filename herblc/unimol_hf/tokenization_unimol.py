# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from transformers import PreTrainedTokenizer
from transformers import logging
import os
import torch
from typing import List, Dict, Optional, Union, Tuple


from .utils.dictionary import Dictionary
from .utils.conformer import ConformerGen
from .utils.padding import pad_1d_tokens, pad_2d, pad_coords

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

class UnimolTokenizer(PreTrainedTokenizer):
    """
    UniMol Tokenizer using the Dictionary class and ConformerGen.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    # define the input names expected by the model (need to match the forward in modeling_unimol.py)
    model_input_names = ["src_tokens", "src_distance", "src_coord", "src_edge_type"]

    def __init__(
        self,
        vocab_file, # must provide
        max_atoms: int = 256,
        remove_hs: bool = True,
        unk_token="[UNK]",
        bos_token="[CLS]",
        eos_token="[SEP]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        dict_dir: Optional[str] = None, # add dict_dir parameter
        **kwargs,
    ):
        _vocab_file = vocab_file
        if dict_dir and not os.path.isabs(_vocab_file):
            _vocab_file = os.path.join(dict_dir, _vocab_file)

        if not os.path.isfile(_vocab_file):
            raise FileNotFoundError(f"Vocabulary file not found at {_vocab_file}")

        self.dictionary = Dictionary.load(_vocab_file)

        for token in [unk_token, bos_token, eos_token, pad_token, mask_token]:
            if token not in self.dictionary:
                logger.warning(f"Special token '{token}' not found in dictionary {_vocab_file}. Adding it.")
                self.dictionary.add_symbol(token)

        kwargs["unk_token"] = unk_token
        kwargs["bos_token"] = bos_token
        kwargs["eos_token"] = eos_token
        kwargs["pad_token"] = pad_token
        kwargs["mask_token"] = mask_token

        super().__init__(**kwargs)
        self.max_atoms = max_atoms
        self.remove_hs = remove_hs
        self.dict_dir = dict_dir if dict_dir else os.path.dirname(_vocab_file)

        self.unk_token_id = self.dictionary.unk()
        self.bos_token_id = self.dictionary.bos()
        self.eos_token_id = self.dictionary.eos()
        self.pad_token_id = self.dictionary.pad()
        # check if mask token exists
        if mask_token in self.dictionary:
            self.mask_token_id = self.dictionary.index(mask_token)
        else:
            logger.error(f"Mask token '{mask_token}' specified but not found in dictionary after loading/adding.")
            self.mask_token_id = self.unk_token_id

        self.conformer_gen = ConformerGen(
            max_atoms=self.max_atoms,
            remove_hs=self.remove_hs,
            dictionary=self.dictionary 
        )

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.dictionary)

    def get_vocab(self) -> Dict[str, int]:
        """Returns the vocabulary as a dictionary."""
        # build vocab from Dictionary object
        return dict(self.dictionary.indices)

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an id using the vocab."""
        return self.dictionary.index(token)

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.dictionary[index]

    def _prepare_unimol_input(self, smiles: str) -> Dict:
        """
        Generates the full UniMol input dictionary for a single SMILES string.
        """
        # get UniMol input
        unimol_input_dict = self.conformer_gen.single_process(smiles)
        return unimol_input_dict

    def _pad_batch(self, batch_inputs) -> Dict[str, np.ndarray]:
        """Pads a batch of UniMol input dictionaries."""
        if isinstance(batch_inputs, list):
            keys = batch_inputs[0].keys()
        elif isinstance(batch_inputs, dict):
            keys = batch_inputs.keys()
        else:
            raise ValueError(f"Invalid batch_inputs type: {type(batch_inputs)}")
        padded_batch = {}

        for key in keys:
            if isinstance(batch_inputs, list):
                values = [torch.tensor(item[key]) for item in batch_inputs] # convert to Tensor for padding
            elif isinstance(batch_inputs, dict):
                values = [torch.tensor(value) for value in batch_inputs[key]]
            else:
                raise ValueError(f"Invalid batch_inputs type: {type(batch_inputs)}")
            if 'src_tokens' in key or 'attention_mask' in key:
                # Pad 1D tokens
                padded_batch[key] = pad_1d_tokens(values, pad_idx=self.pad_token_id).numpy()
            elif 'src_coord' in key:
                # Pad coordinates (assuming pad_idx is 0.0 for coords)
                padded_batch[key] = pad_coords(values, pad_idx=0.0).numpy()
            elif 'src_distance' in key or 'src_edge_type' in key:
                # Pad 2D matrices (distance uses 0.0, edge_type uses pad_token_id)
                pad_value = 0.0 if 'src_distance' in key else self.pad_token_id
                padded_batch[key] = pad_2d(values, pad_idx=pad_value).numpy()
            else:
                logger.warning(f"Unknown key '{key}' during padding. Skipping.")
        
        return padded_batch

    def __call__(
        self,
        smiles: Union[str, List[str]],
        padding: Union[bool, str] = False,
        truncation: bool = False, # Truncation is handled inside ConformerGen
        max_length: Optional[int] = None, # Max length is handled by max_atoms
        return_tensors: Optional[str] = None,
        atoms: Optional[List[str]] = None,
        coordinates: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """
        Main entry point for tokenizing and preparing SMILES strings for UniMol.
        """
        # first check if atoms and coordinates are provided
        if atoms is not None and coordinates is not None:
            # support single or batch
            if isinstance(atoms[0], str):  # compatible with single molecule
                atoms_list = [atoms]
                coordinates_list = [coordinates]
            else:
                atoms_list = atoms
                coordinates_list = coordinates
            batch_inputs = self.conformer_gen.transform_raw(atoms_list, coordinates_list)
        else:
            # original smiles process
            if isinstance(smiles, str):
                batch_smiles = [smiles]
            elif isinstance(smiles, list):
                batch_smiles = smiles
            else:
                raise TypeError("Input must be a string (single SMILES) or a list of strings.")

            # Generate inputs for each SMILES
            batch_inputs = [self._prepare_unimol_input(smiles) for smiles in batch_smiles]
        # Pad the batch if requested
        if padding:
            padded_batch = self._pad_batch(batch_inputs)
            # Add attention mask based on padded src_tokens (input_ids)
            if 'src_tokens' in padded_batch:
                padded_batch['attention_mask'] = (padded_batch['src_tokens'] != self.pad_token_id).astype(int)
                    # Convert to tensors if requested
            if return_tensors == "pt":
                for key, value in padded_batch.items():
                    if key in ['src_tokens', 'src_edge_type', 'attention_mask']:
                        padded_batch[key] = torch.tensor(value, dtype=torch.long)
                    elif key in ['src_distance', 'src_coord']:
                        padded_batch[key] = torch.tensor(value, dtype=torch.float)        
            return padded_batch
        else:   
            # return the non-padded batch data list
            if len(batch_inputs) == 1:
                batch_inputs[0]['attention_mask'] = (batch_inputs[0]['src_tokens'] != self.pad_token_id).astype(int)
                return batch_inputs[0]
            inputs_dict = {}
            for i in range(len(batch_inputs)):
                batch_inputs[i]['attention_mask'] = (batch_inputs[i]['src_tokens'] != self.pad_token_id).astype(int)
                for key, value in batch_inputs[i].items():
                    if key not in inputs_dict:
                        inputs_dict[key] = []
                    inputs_dict[key].append(value)
            return inputs_dict

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Saves the vocabulary specification files (e.g., dictionary file) to a directory.
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        # determine the vocabulary file name
        vocab_file_key = self.vocab_files_names["vocab_file"]
        vocab_file_path = os.path.join(save_directory, f"{filename_prefix + '-' if filename_prefix else ''}{vocab_file_key}")

        try:
            # write the Dictionary content to file
            with open(vocab_file_path, "w", encoding="utf-8") as f:
                for i in range(len(self.dictionary)):
                    symbol = self.dictionary.symbols[i]
                    count = self.dictionary.count[i]
                    f.write(f"{symbol} {count}\n")
            logger.info(f"Vocabulary saved to {vocab_file_path}")
        except Exception as e:
            logger.error(f"Could not save vocabulary file to {vocab_file_path}: {e}")

        return (vocab_file_path,)
