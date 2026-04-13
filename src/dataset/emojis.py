import glob
import numpy as np
import PIL.Image as Image
from pathlib import Path
from grain.sources import RandomAccessDataSource

from .flags import to_onehot


def to_onehot(c, num_classes):
    code = np.zeros((num_classes,), dtype=np.int32)
    code[c] = 1.0
    return code


emojis = [
    'beetle',
    'blossom',
    'blossom_b',
    'bug',
    'butterfly',
    'butterfly_b',
    'butterfly_c',
    'crab',
    'cross',
    'jellyfish',
    'lady_beetle',
    'lizard',
    'lizard_b',
    'lobster',
    'maple_leaf',
    'microbe',
    'mushroom',
    'snowflake',
    'squid',
    'star',
]


emojis_gen = [
    "beetle_blue",
    "beetle_purple",
    "beetle_red",
    "butterfly_b_green",
    "butterfly_b_purple",
    "butterfly_b_red",
    "jellyfish_green",
    "jellyfish_purple",
    "jellyfish_red",
]


class EmojiDataset(RandomAccessDataSource):
    def __init__(self, path, emojis: str | list[str] ='all', size=(40, 40), pad=(8, 8)):
        files = sorted(glob.glob("*.png", root_dir=path))
        if emojis != 'all':
            files = list(filter(
                lambda x: x.split('.')[0] in emojis
                    if isinstance(emojis, list)
                    else x == emojis,
                files
            ))
        self.path = Path(path)
        self.emojis = emojis
        self.files = files
        self.size = size
        self.pad = pad
        self.input_shape = 4, size[0] + 2 * pad[0], size[1] + 2 * pad[1]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = Image.open(self.path / self.files[idx])
        x.thumbnail(self.size, Image.LANCZOS)
        x = np.asarray(x, dtype=np.float32) / 255.0
        x = np.pad(x, (self.pad, self.pad, (0, 0),))
        if x.shape[-1] == 4:
            x[..., :3] = x[..., :3] * x[..., -1:]
        return x.transpose(2, 0, 1), idx
