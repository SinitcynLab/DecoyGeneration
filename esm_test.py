from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Pick a model size (bigger = better, slower). Examples:
# "facebook/esm2_t6_8M_UR50D", "facebook/esm2_t12_35M_UR50D",
# "facebook/esm2_t33_650M_UR50D", "facebook/esm2_t48_15B_UR50D" (requires BF16 & lots of VRAM)
model_name = "facebook/esm2_t33_650M_UR50D"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()

# Your sequence (can have multiple masks)
sequence = "MKTAYIAKQRQISFVKSHFSRQDILDLW<mask>RVRG<mask>AA"

# Canonical 20 amino acids
canonical_aas = list("ACDEFGHIKLMNPQRSTVWY")
aa_ids = tokenizer.convert_tokens_to_ids(canonical_aas)

# Tokenize
inputs = tokenizer(sequence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits  # [1, L, vocab]
probs = torch.softmax(logits, dim=-1)[0]  # [L, vocab]

# Find mask positions
mask_positions = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero().squeeze(-1)

# Report top-5 AA predictions per mask
for i, pos in enumerate(mask_positions.tolist(), start=1):
    # Extract probabilities restricted to canonical AAs
    aa_probs = probs[pos, aa_ids]                    # [20]
    top_vals, top_idx = torch.topk(aa_probs, k=5)    # top-5 within canonical set
    print(f"\nMask #{i} at sequence position {pos}:")
    for p, j in zip(top_vals.tolist(), top_idx.tolist()):
        aa = canonical_aas[j]
        print(f"  {aa}: {p:.4f}")

print("\nFull ESM-2 model card: https://huggingface.co/facebook/esm2_t33_650M_UR50D")