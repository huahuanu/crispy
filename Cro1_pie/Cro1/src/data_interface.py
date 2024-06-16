# basic package
import torchvision.transforms as A
from torch.utils.data import DataLoader
# local package
from data.dataloader import PieDataset
from data.jaad_dataloader import JaadDataset

# global variable
_BATCH_SIZE = 1


class DataInterface():
    def __init__(self, dataset, args):
        super(DataInterface, self).__init__()

        self.data_path = args.data_path
        self.set_path = args.set_path
        self.num_workers = args.num_workers

        self.dataset = dataset

        self.transforms = A.Compose([
            A.ToPILImage(),
            A.RandomPosterize(bits=2),
            A.RandomInvert(p=0.2),
            A.RandomSolarize(threshold=50.0),
            A.RandomAdjustSharpness(sharpness_factor=2),
            A.RandomAutocontrast(p=0.2),
            A.RandomEqualize(p=0.2),
            A.ColorJitter(0.5, 0.3),
            A.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2)),
            A.ToTensor(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage=None):
        if stage == "train" or stage is None:
            self.data_train = PieDataset(
                data_set='train',
                data_path=self.data_path,
                set_path=self.set_path,
                balance=False,
                transforms=self.transforms
            ) if self.dataset == 'PIE' else JaadDataset(
                data_set='train',
                data_path=self.data_path,
                set_path=self.set_path,
                balance=False,
                transforms=self.transforms
            )
            self.data_valid = PieDataset(
                data_set='val',
                data_path=self.data_path,
                set_path=self.set_path,
                balance=False,
                transforms=self.transforms
            ) if self.dataset == 'PIE' else JaadDataset(
                data_set='val',
                data_path=self.data_path,
                set_path=self.set_path,
                balance=False,
                transforms=self.transforms
            )
        if stage == "test" or stage is None:
            self.data_test = PieDataset(
                data_set='test',
                data_path=self.data_path,
                set_path=self.set_path,
                balance=False,
                transforms=self.transforms
            ) if self.dataset == 'PIE' else JaadDataset(
                data_set='test',
                data_path=self.data_path,
                set_path=self.set_path,
                balance=False,
                transforms=self.transforms
            )

    def train_dataloader(self, batch_size=_BATCH_SIZE):
        return DataLoader(
            self.data_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False
        )

    def val_dataloader(self, batch_size=_BATCH_SIZE):
        return DataLoader(
            self.data_valid,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False
        )

    def test_dataloader(self, batch_size=_BATCH_SIZE):
        return DataLoader(
            self.data_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False
        )
