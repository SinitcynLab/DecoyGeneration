import torch
import pandas as pd
import numpy as np

from typing import Iterable, List
from koinapy import Koina
from torch import Tensor

from src.encoders.peptide_encoder import PeptideEncoder
from src.io.peptide_processor import PeptideProcessor

class SpectrumEncoder(PeptideEncoder, PeptideProcessor):
    def __init__(self, special_amino_acids: List[str], charge_const: int = 2, collision_energy_const: int = 25):
        PeptideEncoder.__init__(self)
        PeptideProcessor.__init__(self, special_amino_acids)
        self.charge_const = charge_const
        self.collision_energy_const = collision_energy_const
        self.max_len = 30
        MODEL_ID: str = "Prosit_2019_intensity"
        WEB_ADDRESS: str = "koina.wilhelmlab.org:443"
        self.model = Koina(MODEL_ID, WEB_ADDRESS)

    def get_predicted_spectra(self, sequences: Iterable[str]):
        peptide_list: List[str] = []
        charge_list: List[int] = []
        col_e_list: List[int] = []
        for sequence in sequences:
            for peptide in self.get_all_peptides(sequence):
                if len(peptide) <= self.max_len:
                    start, end = peptide[0], peptide[-1]
                    peptide_list.append(sequence[start:end+1])
                    charge_list.append(self.charge_const)
                    col_e_list.append(self.collision_energy_const)
        
        inputs = pd.DataFrame()
        inputs['peptide_sequences'] = np.array(peptide_list)
        inputs['precursor_charges'] = np.array(charge_list)
        inputs['collision_energies'] = np.array(col_e_list)

        predicted_spectra: pd.DataFrame = self.model.predict(inputs)
        return predicted_spectra

    def __call__(self, sequences: Iterable[str]):
        predicted_spectra = self.get_predicted_spectra(sequences)

        output_list: List[Tensor] = []
        indices = np.unique(predicted_spectra.index)
        for idx in indices:
            peptide_data = predicted_spectra.loc[predicted_spectra.index == idx]
            peptide_list = []
            for _, row in peptide_data.iterrows():
                mz_point = torch.tensor([row['mz'], row['intensities']])
                peptide_list.append(mz_point)
            peptide_tensor = torch.stack(peptide_list, dim=1) # [2, L]
            output_list.append(peptide_tensor)

        return output_list # contains one tensor per peptide, i.e. N peptide-level tensors each with dimension [2, L_n]