import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import ToTensor, Normalize

from hand_datastore import HandsTrackingDataset

from hgan.datasets.utils import ConcatDataset
from hgan.datasets.transforms import Denormalize, CenterCrop, CentroidCrop, Compose, MaskedRandomCrop,

from argparse import ArgumentParser


class CroppedImageDataset(Dataset):
    def __init__(self, hand_view_pairs, ops):
        self.hand_view_pairs = hand_view_pairs
        self.ops = ops
    
    def __getitem__(self, index):
        hand, view = self.hand_view_pairs[index]
        bbox = hand.bounding_box
        assert hand.mask is not None
        cropped_image = view.image[bbox.top:bbox.bottom, bbox.left:bbox.right]
        cropped_mask = hand.mask[bbox.top:bbox.bottom, bbox.left:bbox.right].astype(np.uint8)
        cropped_mask = cropped_mask * 255
        result = dict(rgb=cropped_image, mask=cropped_mask)

        if self.ops:
            result = self.ops(result)

        return result
    
    def __len__(self):
        return len(self.dataset.hand_view_pairs)


class HandDatastoreDataModule(pl.LightningDataModule):
    def __init__(self, dataset_names, hparams):
        self.hparams = hparams
        self.dataset_names = dataset_names
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.denormalize = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    @staticmethod
    def add_model_specific_args(parent_parser, datasets):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--train_bs', type=int, default=8, help='training batch size')
        parser.add_argument('--valid_bs', type=int, default=4, help='validation batch size')
        parser.add_argument('--test_bs', type=int, default=4, help='test batch size')
        parser.add_argument('--num_workers', type=int, default=8, help='# of workers for the dataloader')
        parser.add_argument('--valid_size', type=int, default=1000, help='validation set size')
        parser.add_argument('--test_size', type=int, default=2000, help='test set size')
        parser.add_argument('--train_inp_size', type=int, default=128, help='size of input images for training')
        parser.add_argument('--valid_inp_size', type=int, default=256, help='size of input images for validation')
        parser.add_argument('--shuffle', type=bool, default=True, help='if True, data will be shuffled')
        parser.add_argument('--random_crop', type=bool, default=True, help='if True, random crop the images (for training)')
        return parser

    def prepare_data(self):
        """no need for this"""

    def setup(self):
        self.train_data = []
        self.valid_data = []
        self.test_data = []

        for dataset_name in self.dataset_names:
            train_ops = []
            valid_ops = []
            # Image cropping for the train set
            if self.hparams.random_crop:
                train_ops.append(MaskedRandomCrop(self.hparams.train_inp_size))
            else:
                train_ops.append(CenterCrop(self.hparams.train_inp_size))

            # Image cropping for the valid set
            valid_ops.append(CentroidCrop(self.hparams.valid_inp_size))

            # ToTensor
            train_ops.append(ToTensor())
            valid_ops.append(ToTensor())

            # Normalize
            train_ops.append(Normalize(mean= self.mean, std= self.std))
            valid_ops.append(Normalize(mean= self.mean, std= self.std))
    
            dataset = HandsTrackingDataset(dataset_name)
            hand_view_pairs = dataset.hand_view_pairs
            # Data split into train, valid and test sets.
            indices = list(range(len(dataset.hand_view_pairs)))
            test_pairs = Subset(hand_view_pairs, indices[:self.hparams.test_size])
            valid_pairs = Subset(hand_view_pairs, indices[self.hparams.test_size:self.hparams.test_size + self.hparams.valid_size])
            train_pairs = Subset(hand_view_pairs, indices[self.hparams.test_size + self.hparams.valid_size:])

            self.train_data.append(CroppedImageDataset(train_pairs, Compose(train_ops)))
            self.valid_data.append(CroppedImageDataset(valid_pairs, Compose(valid_ops)))
            self.test_data.append(CroppedImageDataset(test_pairs, Compose(valid_ops)))

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data,
                          batch_size=self.hparams.train_bs,
                          num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.valid_data,
                          batch_size=self.hparams.valid_bs,
                          num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.valid_data,
                          batch_size=self.hparams.test_bs,
                          num_workers=self.hparams.num_workers)
