import random
from torch.utils.data import BatchSampler

class GroupByLengthBatchSampler(BatchSampler):
    """
    Groups samples by their 'length' so that all items in a batch
    have the same context length. This way, no padding is needed.
    """

    def __init__(self, dataset, batch_size, drop_last=True, shuffle_lengths=True):
        """
        Args:
            dataset: Our dataset instance, which must provide get_length(idx).
            batch_size: # of samples per batch, all same length
            drop_last: Whether to drop leftover < batch_size in each length group
            shuffle_lengths: Randomize the order in which length groups are traversed
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle_lengths = shuffle_lengths

        # Build a dict: length -> list of indices
        self.length_to_indices = {}
        for idx in range(len(dataset)):
            item_length = dataset.get_length(idx)
            if item_length not in self.length_to_indices:
                self.length_to_indices[item_length] = []
            self.length_to_indices[item_length].append(idx)

        # Shuffle indices in each length group
        for length in self.length_to_indices:
            random.shuffle(self.length_to_indices[length])

        # Prepare an ordered list of lengths
        self.all_lengths = list(self.length_to_indices.keys())
        if self.shuffle_lengths:
            random.shuffle(self.all_lengths)

    def __iter__(self):
        """
        Yields a list of indices for each batch, all having the same length.
        """
        for length in self.all_lengths:
            indices = self.length_to_indices[length]
            num_full_batches = len(indices) // self.batch_size

            for i in range(num_full_batches):
                start = i * self.batch_size
                end   = start + self.batch_size
                yield indices[start:end]

            remainder = len(indices) % self.batch_size
            if remainder != 0 and not self.drop_last:
                yield indices[-remainder:]

    def __len__(self):
        # Number of batches
        count = 0
        for length in self.length_to_indices:
            group_size = len(self.length_to_indices[length])
            full_batches = group_size // self.batch_size
            count += full_batches
            if (group_size % self.batch_size) != 0 and not self.drop_last:
                count += 1
        return count



class GroupByLengthBatchSampler_Production:
    """
    Groups samples by their 'length' so that all items in a batch
    have the same context length, but returns batches in a random order
    (so consecutive batches can differ in length).
    """

    def __init__(self, dataset, batch_size, drop_last=True):
        """
        Args:
            dataset: Our dataset instance, which must provide get_length(idx).
            batch_size: Number of samples per batch (all with the same length).
            drop_last: Whether to drop leftover < batch_size in each length group.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # 1) Build dict of {length -> list_of_indices}
        self.length_to_indices = {}
        for idx in range(len(dataset)):
            item_length = dataset.get_length(idx)
            if item_length not in self.length_to_indices:
                self.length_to_indices[item_length] = []
            self.length_to_indices[item_length].append(idx)

        # 2) Shuffle indices within each length group
        for length in self.length_to_indices:
            random.shuffle(self.length_to_indices[length])

        # 3) Precompute the total number of batches
        self._num_batches = 0
        for length, indices in self.length_to_indices.items():
            num_full_batches = len(indices) // self.batch_size
            self._num_batches += num_full_batches
            remainder = len(indices) % self.batch_size
            if remainder != 0 and not drop_last:
                self._num_batches += 1

    def __iter__(self):
        """
        Yields a list of sample indices for each batch. Each batch is homogeneous
        in length, but the order of batches across lengths is randomized.
        """
        all_batches = []

        # 4) Collect all possible batches from each length group
        for length, indices in self.length_to_indices.items():
            num_full_batches = len(indices) // self.batch_size

            # full-size batches
            for i in range(num_full_batches):
                start = i * self.batch_size
                end = start + self.batch_size
                all_batches.append(indices[start:end])

            # leftover (partial) batch if drop_last=False
            remainder = len(indices) % self.batch_size
            if remainder != 0 and not self.drop_last:
                all_batches.append(indices[-remainder:])

        # 5) Shuffle the entire list of batches across lengths
        random.shuffle(all_batches)

        # 6) Yield each batch
        for batch_indices in all_batches:
            yield batch_indices

    def __len__(self):
        """Total number of batches per epoch."""
        return self._num_batches