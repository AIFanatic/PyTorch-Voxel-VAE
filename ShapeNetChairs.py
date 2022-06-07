from torch.utils.data import Dataset
from utils.npytar import NpyTarReader
import numpy as np

class ShapeNetChairs(Dataset):
    def __init__(self, filename, input_shape = (1, 32, 32, 32)):
        super().__init__()

        reader = NpyTarReader(filename)
        self.xc = np.zeros((reader.length(), ) + input_shape, dtype = np.float32)
        reader.reopen()

        for ix, (x, name) in enumerate(reader):
            self.xc[ix] = x.astype(np.float32)

    def __len__(self):
        return len(self.xc)

    def __getitem__(self, index):
        x = self.xc[index]
        return 3.0 * x - 1.0