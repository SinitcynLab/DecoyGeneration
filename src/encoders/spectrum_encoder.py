import torch
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft, ifft, fftfreq
from typing import Iterable, List
from koinapy import Koina
from torch import Tensor

from src.encoders.peptide_encoder import PeptideEncoder

from src.proteins.protease import Protease


class SpectrumEncoder(PeptideEncoder):
    def __init__(self, protease: Protease, add_channel: bool = False, charge_const: int = 2, collision_energy_const: int = 30,
                 min_len: int = 8, max_len: int = 30, min_mz = 100, max_mz = 1000, bin_size = 2):
        PeptideEncoder.__init__(self)

        self.protease = protease
        self.charge_const = charge_const
        self.collision_energy_const = collision_energy_const
        self.max_len = max_len
        self.min_len = min_len
        self.max_mz = max_mz
        self.min_mz = min_mz
        self.bin_size = bin_size
        self.vec_len = round((self.max_mz - self.min_mz) / self.bin_size)
        MODEL_ID: str = "Prosit_2019_intensity"
        WEB_ADDRESS: str = "koina.wilhelmlab.org:443"
        self.model = Koina(MODEL_ID, WEB_ADDRESS)

    def get_predicted_spectra(self, sequences: Iterable[str]):
        sequences = [re.sub(r"[UZOBX]", "L", sequence) for sequence in sequences] # Replace 'strange' amino acids by most prevalent (L)
        peptide_list: List[str] = []
        charge_list: List[int] = []
        col_e_list: List[int] = []

        for sequence in sequences:
            for peptide in self.protease.cleave(sequence):
                if self.min_len <= len(peptide.sequence) and len(peptide.sequence) <= self.max_len:
                    peptide_list.append(peptide.sequence)
                    charge_list.append(self.charge_const)
                    col_e_list.append(self.collision_energy_const)
        
        if len(peptide_list) > 0:
            inputs = pd.DataFrame()
            inputs['peptide_sequences'] = np.array(peptide_list)
            inputs['precursor_charges'] = np.array(charge_list)
            inputs['collision_energies'] = np.array(col_e_list)

            predicted_spectra: pd.DataFrame = self.model.predict(inputs, disable_progress_bar=True)

            return predicted_spectra
        else:
            return pd.DataFrame()
        
    def set_tensor_dim(self, peptide_tensor: torch.Tensor):
        peptide_tensor = peptide_tensor.unsqueeze(0) # [1, self.vec_len]
        return peptide_tensor


class VectorSpectrumEncoder(SpectrumEncoder):
    def __call__(self, sequences: Iterable[str]):
        predicted_spectra = self.get_predicted_spectra(sequences)

        boundary_list = np.arange(start=self.min_mz, stop=self.max_mz, step=self.bin_size)
        if predicted_spectra.size > 0:
            output_list: List[Tensor] = []
            indices = np.unique(predicted_spectra.index)
            for idx in indices:
                peptide_data = predicted_spectra.loc[predicted_spectra.index == idx]
                peptide_tensor = torch.zeros(self.vec_len) # [self.vec_len]
                for _, row in peptide_data.iterrows():
                    mz, intensity = row['mz'], row['intensities']
                    if mz < self.min_mz or mz > self.max_mz:
                        continue
                    bin = np.searchsorted(boundary_list, mz, side='right') - 1
                    peptide_tensor[bin] += intensity
                peptide_tensor = (peptide_tensor - peptide_tensor.min()) / (peptide_tensor.max() - peptide_tensor.min()) # normalize (if two intensities were at same ROUNDED m/z)
                output_list.append(self.set_tensor_dim(peptide_tensor)) 

            return output_list # contains one tensor per peptide, i.e. N peptide-level tensors each with dimension [self.vec_len]
        else:
            return None