from pathlib import Path

import pandas as pd


def load_ms2rescore_results(psms_tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(psms_tsv, sep="\t", low_memory=False)
    # Try to find q-value columns
    # Common: q_value, qvalue, q-value, "q-value", "psm_qvalue"
    q_col = None
    for c in ["q_value", "qvalue", "q-value", "qval", "psm_qvalue", "mokapot q-value", "mokapot q_value"]:
        if c in df.columns:
            q_col = c
            break
    if q_col is None:
        # try case-insensitive match
        for c in df.columns:
            if c.lower().replace("_", "").replace("-", "") in ("qvalue", "qval"):
                q_col = c
                break
    if q_col is None:
        raise KeyError(f"Could not find q-value column in MS2Rescore output. Columns: {list(df.columns)[:80]} ...")
    df["_q"] = pd.to_numeric(df[q_col], errors="coerce")

    # Try to find decoy indicator
    # MS2Rescore usually outputs a 'label' or 'is_decoy' or 'isDecoy'
    if "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
    elif "is_decoy" in df.columns:
        df["label"] = df["is_decoy"].apply(lambda x: -1 if bool(x) else 1)
    elif "isDecoy" in df.columns:
        df["label"] = df["isDecoy"].apply(lambda x: -1 if bool(x) else 1)
    else:
        # assume not filtered (dangerous), but keep
        df["label"] = 1

    # peptide sequence column
    pep_col = None
    for c in ["peptide", "sequence", "Peptide", "Peptide sequence", "stripped_peptide"]:
        if c in df.columns:
            pep_col = c
            break
    if pep_col is None:
        # heuristic: any col that contains 'peptide' and not 'mod'
        for c in df.columns:
            if "peptide" in c.lower() and "mod" not in c.lower():
                pep_col = c
                break
    if pep_col is None:
        df["_peptide"] = pd.NA
    else:
        df["_peptide"] = df[pep_col].astype(str)

    # proteins column
    prot_col = None
    for c in ["proteins", "protein", "Protein", "protein_id", "proteinIds"]:
        if c in df.columns:
            prot_col = c
            break
    if prot_col is None:
        df["_proteins"] = ""
    else:
        df["_proteins"] = df[prot_col].astype(str)

    # score column for ranking comparisons
    score_col = None
    for c in ["score", "mokapot score", "mokapot_score", "posterior_error_prob", "posterior_error_probability"]:
        if c in df.columns:
            score_col = c
            break
    if score_col is None:
        df["_score"] = pd.NA
    else:
        df["_score"] = pd.to_numeric(df[score_col], errors="coerce")

    return df
