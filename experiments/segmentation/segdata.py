from torch.utils.data import Dataset, Sampler, DataLoader

class ShapeNetDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Return the file path for the given index
        return idx#, self.file_paths[idx]
