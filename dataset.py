from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=8):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        self.length = {}
        with self.env.begin(write=False) as txn:
            for res in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
                try:
                    self.length[res] = int(txn.get(f'length-{res}'.encode('utf-8')).decode('utf-8'))
                except:
                    pass

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length[self.resolution]

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img
