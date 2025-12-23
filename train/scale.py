from datasets import load_dataset

def sample_parquet(
    input_path: str,
    output_path: str,
    sample_size: int = 1000,
    seed: int = 42,
):
    """
    input_path: Path to a parquet file or a directory containing parquet files
    output_path: Path to the output parquet file
    sample_size: Number of samples to randomly select
    seed: Random seed for reproducibility
    """

    # Load parquet dataset (single file or directory is both supported)
    dataset = load_dataset(
        "parquet",
        data_files=input_path,
        split="train",
    )

    # Shuffle the dataset and take the first `sample_size` examples
    dataset = dataset.shuffle(seed=seed)

    if sample_size < len(dataset):
        dataset = dataset.select(range(sample_size))

    # Save the sampled dataset as a parquet file
    dataset.to_parquet(output_path)

    print(f"Saved {len(dataset)} samples to {output_path}")


if __name__ == "__main__":
    # Example usage
    input_parquet = "wmt_en2tr_6k.parquet"     # or "data/parquet_dir/"
    output_parquet = "wmt_en2tr_1k.parquet"

    sample_parquet(
        input_path=input_parquet,
        output_path=output_parquet,
        sample_size=1000,
        seed=42,
    )
