from decoygen.generate import main

if __name__ == "__main__":
    # Example invocation mirroring training_model.py style.
    # Adjust paths and parameters as needed.
    main([
        "--checkpoint", "runs/exp1/final.pt",
        "--num", "200",
        "--out", "decoys.fasta",
        "--temperature", "1.0",
        "--top-p", "0.9",
        "--repetition-penalty", "1.2",
        "--min-length", "7",
        "--max-length", "64",
        "--mass-min", "500.0",
        "--mass-max", "3000.0",
        "--max-missed-cleavages", "2",
        "--reject-identity", "0.8",
        "--device", "mps"  # Use "mps" for Apple Silicon
    ])
