from .aminoacids import AMINOACIDS

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import List, Set


@dataclass
class Peptide:
    # Sequence of the peptide
    sequence: str
    # Which replacements are allowed for every position in order
    # not to break the protease cleavage rules
    allowed_replacements: List[Set[str]]
    # Range of positions where most of the modifications are allowed (for example, for trypsin it is everything but the last amino acid).
    # Note, that even for these positions not all amino acid replacements may be allowed but most of the modifications are allowed here.
    flexible_range: range = None


class Protease(ABC):
    @abstractmethod
    def cleave(self, sequence: str) -> List[Peptide]:
        pass


class TerminusProtease(Protease):
    def __init__(
        self,
        # Which side of the cleavage site is the cleavage residue on? "N" or "C"
        cleavage_side: str,
        # Which residues are present at the cleavage site? For example, for trypsin it would be {"K", "R"}
        cleavage_residues: Set[str],
        # Which residues are not allowed to follow the cleavage site? For example, for trypsin it would be {"P"}
        blocked_residues: Set[str] | None = None,
    ):
        # Notation:
        #  - n_allowed: which amino acids should be present at the n terminus of the cleavage site
        #  - n_denied: which amino acids should not be present at the n terminus of the cleavage site
        #  - c_allowed: which amino acids should be present at the c terminus of the cleavage site
        #  - c_denied: which amino acids should not be present at the c terminus of the cleavage site
        # In order for cleavage to happen, one of the "allow" rules should be satisfied and none of the "deny" rules should be violated 
        if cleavage_side == "C":
            self.n_allowed = set()
            self.n_denied = blocked_residues or set()
            self.c_allowed = cleavage_residues
            self.c_denied = set()
        elif cleavage_side == "N":
            self.n_allowed = cleavage_residues
            self.n_denied = set()
            self.c_allowed = set()
            self.c_denied = blocked_residues or set()
        else:
            raise ValueError(f"Invalid cleavage side: {cleavage_side}. Should be either 'N' or 'C'.")
        
        self.cleavage_side = cleavage_side

    def cleave(self, sequence: str) -> List[Peptide]:
        peptides = []
        current_sequence = ""
        for i in range(len(sequence)):
            current_sequence += sequence[i]

            # Do we cleave after the current position?
            if i == len(sequence) - 1:
                cleave = True
            else:
                n_residue = sequence[i]
                c_residue = sequence[i + 1]
                cleave = (
                    (n_residue in self.n_allowed or c_residue in self.c_allowed)
                    and n_residue not in self.n_denied
                    and c_residue not in self.c_denied
                )
            
            if cleave and current_sequence:
                allowed_replacement_list = []
                for j in range(len(current_sequence)):
                    denied_replacements = set()
                    allowed_replacements = set(AMINOACIDS)
                    if j == 0:
                        # N terminus of peptide and C terminus of cleavage site.
                        # n_allowed are denied here in order not to create a new cleavage site *after* the current position.
                        # n_denied are allowed here since they will not break *existing* cleavage site.
                        # c_allowed are allowed here since they will provoke already existing cleavage site.
                        # c_denied are denied here since they will break *existing* cleavage site.
                        denied_replacements = denied_replacements.union(self.n_allowed)
                        denied_replacements = denied_replacements.union(self.c_denied)

                        # If protease forces presence of certain amino acids at the cleavage C terminus (peptide N terminus),
                        # then we must also force presence of these amino acids at the peptide N terminus in order not to break cleavage site.
                        if self.c_allowed:
                            allowed_replacements = allowed_replacements.intersection(self.c_allowed)
                    if j == len(current_sequence) - 1:
                        # C terminus of peptide and N terminus of cleavage site.
                        # n_allowed are allowed here since they will provoke already existing cleavage site.
                        # n_denied are denied here since they will break *existing* cleavage site.
                        # c_allowed are denied here in order not to create a new cleavage site *before* the current position.
                        # c_denied are allowed here since they will not break *existing* cleavage site.
                        denied_replacements = denied_replacements.union(self.n_denied)
                        denied_replacements = denied_replacements.union(self.c_allowed)

                        # If protease forces presence of certain amino acids at the cleavage N terminus (peptide C terminus),
                        # then we must also force presence of these amino acids at the peptide C terminus in order not to break cleavage site.
                        if self.n_allowed:
                            allowed_replacements = allowed_replacements.intersection(self.n_allowed)
                    if j != 0 and j != len(current_sequence) - 1:
                        # Internal position of the peptide. Both n_allowed and c_allowed are denied here since they
                        # may introduce a new cleavage. Both n_denied and c_denied are allowed here since they will not break *existing*
                        # cleavage sites.
                        denied_replacements = denied_replacements.union(self.n_allowed)
                        denied_replacements = denied_replacements.union(self.c_allowed)

                    allowed_replacements = allowed_replacements - denied_replacements
                    # The set may be empty in some weird cases (for example, a single amino acid peptide at the end of the protein with trypsin),
                    # but we can forcefully allow current amino acid in this case.
                    allowed_replacements.add(current_sequence[j])
                    allowed_replacement_list.append(allowed_replacements)

                if len(current_sequence) == 1:
                    # Peptides of length 1 are a bit special case so let's not call them flexible.
                    flexible_range = range(0, 0)
                elif self.cleavage_side == "N":
                    flexible_range = range(0, len(current_sequence) - 1)
                else:
                    assert self.cleavage_side == "C"
                    flexible_range = range(1, len(current_sequence))

                peptides.append(Peptide(current_sequence, allowed_replacement_list, flexible_range))
                current_sequence = ""

        return peptides
    

PROTEASES = {
    "trypsin": TerminusProtease(
        cleavage_side="N",
        cleavage_residues={"K", "R"},
    ),
    "trypsin_p": TerminusProtease(
        cleavage_side="N",
        cleavage_residues={"K", "R"},
        blocked_residues={"P"},
    ),
    "chymotrypsin": TerminusProtease(
        cleavage_side="N",
        cleavage_residues={"F", "W", "Y"},
        blocked_residues={"P"},
    ),
    "pepsin": TerminusProtease(
        cleavage_side="N",
        cleavage_residues={"F", "W", "Y", "L"},
    ),
    "aspn": TerminusProtease(
        cleavage_side="C",
        cleavage_residues={"D"},
    ),
    "gluc": TerminusProtease(
        cleavage_side="N",
        cleavage_residues={"E"},
    ),
    "lysc": TerminusProtease(
        cleavage_side="N",
        cleavage_residues={"K"},
    ),
    "argc": TerminusProtease(
        cleavage_side="N",
        cleavage_residues={"R"},
    ),
}

def list_proteases() -> List[str]:
    return list(PROTEASES.keys())

def get_protease(protease_name: str) -> Protease:
    if protease_name not in PROTEASES:
        raise ValueError(f"Protease {protease_name} is not supported. Supported proteases are: {list_proteases()}")
    return PROTEASES[protease_name]
