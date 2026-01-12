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
from src.io.peptide_processor import PeptideProcessor

class SpectrumEncoder(PeptideEncoder, PeptideProcessor):
    def __init__(self, special_amino_acids: List[str], add_channel: bool = False, charge_const: int = 2, collision_energy_const: int = 30,
                 min_len: int = 8, max_len: int = 30, max_mz = 4000):
        PeptideEncoder.__init__(self)
        PeptideProcessor.__init__(self, special_amino_acids)
        self.charge_const = charge_const
        self.collision_energy_const = collision_energy_const
        self.max_len = max_len
        self.min_len = min_len
        self.max_mz = max_mz
        self.add_channel = add_channel
        MODEL_ID: str = "Prosit_2019_intensity"
        WEB_ADDRESS: str = "koina.wilhelmlab.org:443"
        self.model = Koina(MODEL_ID, WEB_ADDRESS)

    def get_predicted_spectra(self, sequences: Iterable[str]):
        sequences = [re.sub(r"[UZOBX]", "L", sequence) for sequence in sequences] # Replace 'odd' amino acids by most prevalent (L)
        peptide_list: List[str] = []
        charge_list: List[int] = []
        col_e_list: List[int] = []
        for sequence in sequences:
            for peptide in self.get_all_peptides(sequence):
                if self.min_len <= len(peptide) and len(peptide) <= self.max_len:
                    start, end = peptide[0], peptide[-1]
                    peptide_list.append(sequence[start:end+1])
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
        if self.add_channel:
            peptide_tensor = peptide_tensor.reshape((1,1,self.max_mz)) # [1, 1, self.max_mz]
        else:
            peptide_tensor = peptide_tensor.unsqueeze(0) # [1, self.max_mz]
        return peptide_tensor


class VectorSpectrumEncoder(SpectrumEncoder):
    def __call__(self, sequences: Iterable[str]):
        predicted_spectra = self.get_predicted_spectra(sequences)

        if predicted_spectra.size > 0:
            output_list: List[Tensor] = []
            indices = np.unique(predicted_spectra.index)
            for idx in indices:
                peptide_data = predicted_spectra.loc[predicted_spectra.index == idx]
                peptide_tensor = torch.zeros(self.max_mz) # [self.max_mz]
                for _, row in peptide_data.iterrows():
                    mz, intensity = row['mz'], row['intensities']
                    rounded_mz = round(mz)
                    if rounded_mz > self.max_mz:
                        continue # truncate at max_mz
                    peptide_tensor[rounded_mz] += intensity
                min_intensity = peptide_tensor.min()
                max_intensity = peptide_tensor.max()
                peptide_tensor = (peptide_tensor - min_intensity) / (max_intensity - min_intensity) # normalize (if two intensities were at same ROUNDED m/z)
                output_list.append(self.set_tensor_dim(peptide_tensor)) 

            return output_list # contains one tensor per peptide, i.e. N peptide-level tensors each with dimension [self.max_mz]
        else:
            return None
        
class SmoothVectorSpectrumEncoder(VectorSpectrumEncoder):
    def __init__(self, special_amino_acids: List[str], add_channel: bool = False, charge_const: int = 2, collision_energy_const: int = 25,
                 min_len: int = 8, max_len: int = 30, max_mz = 4000, window_size: int = 100):
        VectorSpectrumEncoder.__init__(self, special_amino_acids, add_channel, charge_const, collision_energy_const,
                 min_len, max_len, max_mz)
        self.window_size = window_size

    def smooth(self, y: np.ndarray, window_size: int):
        half_kernel = np.arange(0, window_size//2)
        half_kernel = half_kernel / np.sum(half_kernel)
        kernel = np.concatenate((half_kernel, np.flip(half_kernel)), axis=0)
        y_smooth = np.convolve(y, kernel, mode='same')
        return np.expand_dims(y_smooth, axis=0)

    def __call__(self, sequences: Iterable[str]):
        output_list = VectorSpectrumEncoder.__call__(self, sequences)

        for i, peptide_tensor in enumerate(output_list):
            y = peptide_tensor.numpy().squeeze()
            y_smooth = self.smooth(y, self.window_size)
            #plt.plot(np.arange(4000), y_smooth.squeeze())
            #plt.show(block=True)
            output_list[i] = self.set_tensor_dim(torch.Tensor(y_smooth))

        return output_list
    
class TupleSpectrumEncoder(SpectrumEncoder):
    def __call__(self, sequences: Iterable[str]):
        predicted_spectra = self.get_predicted_spectra(sequences)

        if predicted_spectra.size > 0:
            output_list: List[Tensor] = []
            indices = np.unique(predicted_spectra.index)
            for peptide_idx in indices:
                peptide_data = predicted_spectra.loc[predicted_spectra.index == peptide_idx]
                peptide_data = peptide_data.sort_values(by='mz', ascending=True, inplace=False)
                peptide_tensor = torch.zeros((len(peptide_data), 2)) # [2, L_idx]
                mz_idx = 0
                for _, row in peptide_data.iterrows():
                    mz, intensity = row['mz'], row['intensities']
                    peptide_tensor[mz_idx,0] = mz
                    peptide_tensor[mz_idx,1] = intensity
                    mz_idx += 1
                peptide_tensor[:, 0] = (peptide_tensor[:, 0] - peptide_tensor[:, 0].min()) / (peptide_tensor[:, 0].max() - peptide_tensor[:, 0].min()) # normalize
                output_list.append(self.set_tensor_dim(peptide_tensor)) 

            return output_list # contains one tensor per peptide, each with dim [2, L_idx] where L_idx depends on the peptide length
        else:
            return None

class FFTSpectrumEncoder(VectorSpectrumEncoder):
    def __init__(self, special_amino_acids, add_channel = False, charge_const = 2, 
                 collision_energy_const = 25, min_len = 8, max_len = 30, max_mz=4000, sample_rate = 500, threshold = 10):
        VectorSpectrumEncoder.__init__(self, special_amino_acids, add_channel, charge_const, collision_energy_const, min_len, max_len, max_mz)
        self.sample_rate = sample_rate
        self.threshold = threshold

    def __call__(self, sequences: Iterable[str]):
        output_list = VectorSpectrumEncoder.__call__(self, sequences)

        for i, peptide_tensor in enumerate(output_list):
            y = peptide_tensor.numpy().squeeze()
            
            # get fft and amplitudes of each frequency:
            transformed_y = fft(y)
            frequencies = fftfreq(len(y), d=1/self.sample_rate)

            # filter out weak frequencies:
            transformed_y[np.abs(frequencies) > self.threshold] = 0

            # plot (for debugging):
            _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))
            horizontal_axis = np.arange(start=0, stop=4000, step=1)
            ax1.plot(horizontal_axis, y, label="original")
            ax1.plot(horizontal_axis, transformed_y, label="filtered")
            ax1.legend()
            ax2.plot(frequencies, np.abs(transformed_y))
            plt.show()

            output_list[i] = self.set_tensor_dim(torch.tensor(transformed_y))

        return output_list