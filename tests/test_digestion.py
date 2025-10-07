from decoygen.prepare_data import tryptic_cleave, generate_peptides

def test_tryptic_cleave_basic():
    prot = "MPEPTIDEKRALP"
    cuts = tryptic_cleave(prot)
    # Ensure start and end present
    assert cuts[0] == 0 and cuts[-1] == len(prot)
    # K/R cleavage positions except when followed by P
    # K at index 8, R at 9, but R followed by A so K and R both cleavage sites
    assert 9 in cuts and 10 in cuts  # positions after residues

def test_generate_peptides_missed():
    prot = "ACDKRQL"  # K R contiguous then Q L
    peps0 = generate_peptides(prot, max_missed=0, min_len=1, max_len=50)
    peps1 = generate_peptides(prot, max_missed=1, min_len=1, max_len=50)
    # Allowing one missed should produce at least as many peptides
    assert len(peps1) >= len(peps0)

def test_length_filter():
    prot = "ACDEKRF"  # length 7
    peps = generate_peptides(prot, max_missed=0, min_len=7, max_len=7)
    assert all(len(p)==7 for p in peps)