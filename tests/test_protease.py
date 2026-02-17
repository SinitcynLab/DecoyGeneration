from src.proteins.protease import get_protease
from src.proteins.aminoacids import AMINOACIDS

AA = set(AMINOACIDS)


def test_trypsin_p():
    peptides = get_protease("trypsin_p").cleave("AAKPARG")
    assert len(peptides) == 2
    assert peptides[0].sequence == "AAKPAR"
    assert peptides[1].sequence == "G"
    # First peptide: pos 0 denies {K, R, P}, pos 1-4 deny {K, R}, pos 5 (C-term) must be K or R
    assert peptides[0].allowed_replacements[0] == AA - {"K", "R", "P"}
    for i in range(1, 5):
        if i == 2:
            # K is typically forbidden in the middle of the peptide, but in this case it is
            # the original amino acid.
            assert peptides[0].allowed_replacements[i] == AA - {"R"}
        else:
            assert peptides[0].allowed_replacements[i] == AA - {"K", "R"}
    assert peptides[0].allowed_replacements[5] == {"K", "R"}
    assert peptides[1].allowed_replacements[0] == {"G"}


def test_trypsin():
    p1 = get_protease("trypsin").cleave("AGKA")
    assert [p.sequence for p in p1] == ["AGK", "A"]
    p2 = get_protease("trypsin").cleave("AKRA")
    assert [p.sequence for p in p2] == ["AK", "R", "A"]


def test_chymotrypsin():
    p = get_protease("chymotrypsin").cleave("AGFAL")
    assert [x.sequence for x in p] == ["AGF", "AL"]


def test_aspn():
    peptides = get_protease("aspn").cleave("AGDALG")
    assert len(peptides) == 2
    assert peptides[0].sequence == "AG"
    assert peptides[1].sequence == "DALG"
    assert peptides[0].allowed_replacements[0] == {"A", "D"}
    assert peptides[0].allowed_replacements[1] == AA - {"D"}
    assert peptides[1].allowed_replacements[0] == {"D"}
    for i in range(1, 4):
        assert peptides[1].allowed_replacements[i] == AA - {"D"}


def test_gluc():
    p = get_protease("gluc").cleave("AGEAL")
    assert [x.sequence for x in p] == ["AGE", "AL"]


def test_lysc():
    p = get_protease("lysc").cleave("AGKAL")
    assert [x.sequence for x in p] == ["AGK", "AL"]


def test_argc():
    p = get_protease("argc").cleave("AGRAL")
    assert [x.sequence for x in p] == ["AGR", "AL"]
