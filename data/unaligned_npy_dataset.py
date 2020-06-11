import os.path
from data.base_dataset import BaseDataset
# from data.image_folder import make_dataset
import numpy as np
import random
import torchvision.transforms as transforms

NPY_EXTENSIONS = ['.np', '.npy', '.NP', '.NPY']


def is_npy_file(filename):
    return any(filename.endswith(extension) for extension in NPY_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_npy_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class UnalignedNpyDataset(BaseDataset):
    """
    This dataset class loads unaligned/unpaired datasets of .npy files, which must be  1-channel (2 dimensional) arrays,
    all with exactly the same shape (w,h)

    It requires two directories to host training arrays from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load arrays from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load arrays from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an array in the input domain
            B (tensor)       -- its corresponding array in the target domain
            A_paths (str)    -- array paths
            B_paths (str)    -- array paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A = np.load(A_path)
        B = np.load(B_path)

	# Change to 3D single channel
        A = transforms.ToTensor(A)
        B = transforms.ToTensor(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of arrays in the dataset.

        As we have two datasets with potentially different number of arrays,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
