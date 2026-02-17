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
from src.encoders.spectrum_encoder import VectorSpectrumEncoder
        
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
    def __init__(self, special_amino_acids, add_channel = False, charge_const = 2, collision_energy_const = 30, 
                 min_len = 8, max_len = 30, min_mz=100, max_mz=1000, normalize_mz = True):
        super().__init__(special_amino_acids, add_channel, charge_const, collision_energy_const, min_len, max_len, min_mz, max_mz)
        self.normalize_mz = normalize_mz

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
                if self.normalize_mz:
                    peptide_tensor[:, 0] = (peptide_tensor[:, 0] - peptide_tensor[:, 0].min()) / (peptide_tensor[:, 0].max() - peptide_tensor[:, 0].min()) # normalize
                output_list.append(self.set_tensor_dim(peptide_tensor)) 

            return output_list # contains one tensor per peptide, each with dim [2, L_idx] where L_idx depends on the peptide length
        else:
            return None

class FFTSpectrumEncoder(VectorSpectrumEncoder):
    def __init__(self, special_amino_acids, add_channel = False, charge_const = 2, 
                 collision_energy_const = 25, min_len = 8, max_len = 30, min_mz = 100, max_mz = 1000, sample_rate = 500, threshold = 10):
        VectorSpectrumEncoder.__init__(self, special_amino_acids, add_channel, charge_const, collision_energy_const, min_len, max_len, min_mz, max_mz)
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
            
            # get back the filtered signal from the transformed one:
            filtered_y = ifft(transformed_y)

            output_list[i] = self.set_tensor_dim(torch.tensor(filtered_y.real))

        return output_list