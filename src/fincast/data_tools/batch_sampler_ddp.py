import torch
from torch.utils.data import Sampler

class GroupByLengthBatchSampler_DDP(Sampler):
    """
    Groups samples by length and partitions them across different DDP ranks.
    All shuffling is done *inside* __iter__, with a single torch.Generator,
    ensuring identical shuffling on all ranks.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        drop_last=True,
        num_replicas=None,
        rank=None,
        seed=5
    ):
        """
        Args:
            dataset: dataset with a get_length(idx) method.
            batch_size: number of items per batch (homogeneous length).
            drop_last: whether to drop leftover < batch_size in each length group.
            num_replicas: total number of DDP processes (world_size).
            rank: current process index among [0..num_replicas-1].
            seed: base random seed for generating random ordering.
        """
        super().__init__()
        
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available.")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available.")
            rank = torch.distributed.get_rank()

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0  # Will get updated via set_epoch

        # Build dict of {length -> list_of_indices}, but do NOT shuffle here.
        # We simply store them in the original order. We'll shuffle inside __iter__.
        self.length_to_indices = {}
        for idx in range(len(dataset)):
            length = dataset.get_length(idx)
            if length not in self.length_to_indices:
                self.length_to_indices[length] = []
            self.length_to_indices[length].append(idx)

        # Precompute the total number of batches (over all ranks, combined).
        self._num_batches = 0
        for length, indices in self.length_to_indices.items():
            num_full_batches = len(indices) // self.batch_size
            self._num_batches += num_full_batches
            remainder = len(indices) % self.batch_size
            if remainder != 0 and not self.drop_last:
                self._num_batches += 1

    def __iter__(self):
        """
        Yields a list of sample indices for each batch (only the subset for this rank).
        """
        # Create a generator seeded identically across all ranks for this epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # 1) For each length group, shuffle the indices with the same generator
        #    so that all ranks get the same "local" shuffle.
        all_batches = []
        for length, indices in self.length_to_indices.items():
            # Copy to avoid modifying the original stored list
            shuffled_indices = indices[:]  # shallow copy
            # Use torch.randperm to shuffle consistently across ranks
            perm = torch.randperm(len(shuffled_indices), generator=g).tolist()
            shuffled_indices = [shuffled_indices[i] for i in perm]

            # Build full-size batches
            num_full_batches = len(shuffled_indices) // self.batch_size
            for i in range(num_full_batches):
                start = i * self.batch_size
                end = start + self.batch_size
                all_batches.append(shuffled_indices[start:end])

            # Possible leftover
            remainder = len(shuffled_indices) % self.batch_size
            if remainder != 0 and not self.drop_last:
                all_batches.append(shuffled_indices[-remainder:])

        # 2) Shuffle the entire list of batches with the same generator
        perm = torch.randperm(len(all_batches), generator=g).tolist()
        all_batches = [all_batches[i] for i in perm]

        # 3) Partition these batches across the DDP ranks.
        #    By default, we'll "drop" leftover so that each rank sees the
        #    same number of batches (common approach for distributed).
        total_batches = len(all_batches)
        batches_per_replica = total_batches // self.num_replicas
        total_used = batches_per_replica * self.num_replicas

        start_idx = self.rank * batches_per_replica
        end_idx = start_idx + batches_per_replica
        my_batches = all_batches[start_idx:end_idx]

        # 4) Yield the subset for this rank
        for batch_indices in my_batches:
            yield batch_indices

    def __len__(self):
        """
        Returns the number of batches (per replica).
        Because we drop remainder in the final partitioning,
        each replica sees the same count: floor_div(total_batches, num_replicas).
        """
        return self._num_batches // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch. Must be called at the beginning of each epoch
        in DDP to ensure a different shuffle sequence each epoch.
        """
        self.epoch = epoch
