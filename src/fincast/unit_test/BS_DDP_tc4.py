import torch
from collections import defaultdict
from fincast.data_tools.batch_sampler_ddp import GroupByLengthBatchSampler_DDP

# Mock dataset with custom lengths
class MockDataset:
    def __init__(self, lengths):
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)

    def get_length(self, idx):
        return self.lengths[idx]

# Helper functions for checking correctness
def check_consistency_across_ranks(batches_per_rank):
    all_batches = [tuple(batch) for rank_batches in batches_per_rank for batch in rank_batches]
    unique_batches = set(all_batches)
    assert len(all_batches) == len(unique_batches), "Batches overlap between ranks!"

def replicate_sampler_logic(dataset, batch_size, drop_last, num_replicas, epoch, base_seed):
    """
    Produce the EXACT set of indices used by GroupByLengthBatchSampler_DDP,
    given the same hyperparameters and epoch.
    """
    g = torch.Generator()
    g.manual_seed(base_seed + epoch)

    # 1) Indices grouped by length in insertion order
    length_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        length = dataset.get_length(idx)
        length_to_indices[length].append(idx)

    # 2) Shuffle within each length group, form batches
    all_batches = []
    for length, indices in length_to_indices.items():
        # Copy to avoid modifying the original list
        shuffled_indices = indices[:]
        perm = torch.randperm(len(shuffled_indices), generator=g).tolist()
        shuffled_indices = [shuffled_indices[i] for i in perm]

        # Full-size batches
        num_full_batches = len(shuffled_indices) // batch_size
        for i in range(num_full_batches):
            start = i * batch_size
            end = start + batch_size
            all_batches.append(shuffled_indices[start:end])

        # Possible leftover
        remainder = len(shuffled_indices) % batch_size
        if remainder > 0 and not drop_last:
            all_batches.append(shuffled_indices[-remainder:])

    # 3) Shuffle entire list of batches
    perm = torch.randperm(len(all_batches), generator=g).tolist()
    all_batches = [all_batches[i] for i in perm]

    # 4) Drop leftover batches so total is divisible by num_replicas
    total_batches = len(all_batches)
    used_batches = all_batches[: (total_batches // num_replicas) * num_replicas]

    # Return as set of indices
    final_indices = set(idx for batch in used_batches for idx in batch)
    return final_indices


def check_full_coverage(batches_per_rank, dataset, drop_last, batch_size, num_replicas, epoch=0, base_seed=42):
    """
    Compare the actual sampled indices from each rank to the EXACT indices
    that replicate_sampler_logic() would produce.
    """
    # Indices actually used by the sampler across all ranks
    sampled_indices = set(idx for rank_batches in batches_per_rank
                                for batch in rank_batches
                                for idx in batch)

    # Indices the sampler *should* use
    expected_indices = replicate_sampler_logic(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        num_replicas=num_replicas,
        epoch=epoch,
        base_seed=base_seed
    )

    missing = expected_indices - sampled_indices
    extra = sampled_indices - expected_indices

    if missing:
        print(f"⚠️ Missing indices: {missing}")
    if extra:
        print(f"⚠️ Extra indices: {extra}")

    assert not missing, f"Missing indices: {missing}"
    assert not extra, f"Extra indices: {extra}"

    print(f"✅ Full coverage: {len(expected_indices)} indices used exactly as expected.")

def check_batch_size_and_length(batches, dataset):
    for batch in batches:
        lengths = {dataset.get_length(idx) for idx in batch}
        assert len(lengths) == 1, f"Multiple lengths found in one batch!"

# Simulate the sampler across multiple ranks
def simulate_ddp_sampler(dataset, batch_size, num_replicas, drop_last=True, seed=42, epochs=2):
    print("Dataset:")
    for idx, length in enumerate(dataset.lengths):
        print(f"Idx {idx}: length={length}")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch}:")
        batches_per_rank = []
        for rank in range(num_replicas):
            sampler = GroupByLengthBatchSampler_DDP(
                dataset=dataset,
                batch_size=batch_size,
                drop_last=drop_last,
                num_replicas=num_replicas,
                rank=rank,
                seed=seed
            )
            sampler.set_epoch(epoch)
            rank_batches = list(sampler)
            batches_per_rank.append(rank_batches)
            print(f"Rank {rank} batches: {rank_batches}")

            # Check batch size and lengths
            check_batch_size_and_length(rank_batches, dataset)

        # Check consistency and coverage
        check_consistency_across_ranks(batches_per_rank)
        check_full_coverage(batches_per_rank, dataset=dataset, 
                            drop_last=drop_last, batch_size=batch_size, num_replicas=num_replicas,
                            epoch=epoch, base_seed=seed)

# Example usage
if __name__ == "__main__":
    # Create a mock dataset with varying lengths
    lengths = [64]*10 + [128]*12 + [256]*8 + [512]*6  # 36 samples total
    dataset = MockDataset(lengths)

    # Test parameters
    batch_size = 4
    num_replicas = 4
    drop_last = True
    seed = 42

    # Run the simulation for multiple epochs
    simulate_ddp_sampler(dataset, batch_size, num_replicas, drop_last=drop_last, epochs=3, seed=seed)
