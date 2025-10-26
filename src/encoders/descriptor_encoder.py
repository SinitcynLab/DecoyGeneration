import torch

from src.encoders.peptide_encoder import PeptideEncoder
from peptidy.descriptors import compute_descriptors
from typing import Iterable

class DescriptionEncoder(PeptideEncoder):
    def __init__(self, features: list[str] = None, pH : float = 7):
        if features is None:
            features = [
                'aliphatic_index', 
                'aminoacid_frequencies', 
                'aromaticity', 
                'average_number_rotatable_bonds', 
                'charge', 
                'charge_density', 
                'energy_based_on_logP', 
                'hydrophobic_aa_ratio', 
                'instability_index', 
                'isoelectric_point', 
                'length', 
                'molecular_weight', 
                'n_h_donors', 
                'n_h_acceptors', 
                'topological_polar_surface_area']
        super().__init__()
        self.features = features
        self.pH = pH
        self.get_features = lambda x : compute_descriptors(x, self.features, self.pH)

    def __call__(self, sequences : Iterable[str]) -> torch.Tensor:
        dict_list = list(map(self.get_features, sequences)) # will be list of dicts
        n_features = len(dict_list[0].values()) # hacky, could be moved into __init__
        x = torch.zeros((len(sequences), n_features))
        for i, dictionary in enumerate(dict_list):
            value_list = list(dictionary.values())
            x[i,:] = torch.FloatTensor(value_list)
        print(x[1:10])
        x = self.normalize_tensor(x)
        return x