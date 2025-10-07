from decoygen.train import main

if __name__ == "__main__":
    main([
        "--data", "data/peptides.txt",
        "--out-dir", "runs/exp1",
        "--steps", "2000",
        "--batch-size", "64",
        "--d-model", "256",
        "--layers", "6",
        "--heads", "8",
        "--ff", "1024",
        "--lr", "3e-4",
        "--warmup", "200",
        "--label-smoothing", "0.1",
        "--device", "mps"  # Use "mps" for Apple Silicon (M chips)
    ])