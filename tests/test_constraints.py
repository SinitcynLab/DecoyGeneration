from decoygen.constraints import count_missed_cleavages

def test_no_missed_simple():
    assert count_missed_cleavages(list("ACDE")) == 0

def test_single_terminal_kr_not_missed():
    assert count_missed_cleavages(list("ACDK")) == 0

def test_internal_kr_missed():
    # Internal K followed by A counts as missed cleavage
    assert count_missed_cleavages(list("AKAC")) == 1

def test_multiple_missed():
    assert count_missed_cleavages(list("AKARAC")) >= 2  # depending on logic counts 2 internal sites
