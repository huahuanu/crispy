# basic package
import torch
import os
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
# local package
from src.data_interface import DataInterface
from src.model_interface import ModelInterface


class TrainInterface():
    def __init__(self, dataset, args):
        super(TrainInterface, self).__init__()
        # Check args is None or not
        if args is None:
            raise ValueError("args is None.")
        self.seed_all(args.seed)

        self.data_loader = DataInterface(dataset=dataset, args=args) if not args.debug else None
        self.model = ModelInterface(args=args)

        self.optim = torch.optim.Adam(self.model.model.parameters(), lr=args.lr)
        self.criterion = torch.nn.BCELoss()
        self.reg_criterion = torch.nn.MSELoss()

        model_folder_name = args.save_name
        self.checkpoint_filepath = 'checkpoints/{}.pt'.format(model_folder_name)
        self.writer = SummaryWriter(log_dir='logs/{}'.format(model_folder_name))

    def run(self, args):
        # Check mode
        if not args.train and not args.test:
            raise ValueError("args.train and args.test are both False. You must choose not less one of them.")
        # Instantiate Data objects
        if args.train:
            if not args.debug:
                self.data_loader.setup(stage='train')
                train_data, val_data = self.data_loader.train_dataloader(
                    batch_size=args.batch_size), self.data_loader.val_dataloader(batch_size=args.batch_size)
            else:
                train_data = [[torch.randn(size=(args.batch_size, args.bbox_size, args.time_scale)),
                               torch.randn(size=(args.batch_size, args.vel_size, args.time_scale)),
                               torch.tensor([[1] * args.batch_size])]]
                val_data = [[torch.randn(size=(args.batch_size, args.bbox_size, args.time_scale)),
                             torch.randn(size=(args.batch_size, args.vel_size, args.time_scale)),
                             torch.tensor([[1] * args.batch_size])]]
            # Instantiate Model objects
            self.model.train(
                train_data, val_data,
                criterion=self.criterion,
                optimizer=self.optim,
                checkpoint_filepath=self.checkpoint_filepath,
                writer=self.writer,
                args=args
            )

        if args.test:
            if not args.debug:
                self.data_loader.setup(stage='test')
                test_data = self.data_loader.test_dataloader()
            else:
                test_data = [[torch.randn(size=(args.batch_size, args.bbox_size, args.time_scale)),
                              torch.randn(size=(args.batch_size, args.vel_size, args.time_scale)),
                              torch.tensor([[1] * args.batch_size])],
                             [torch.randn(size=(args.batch_size, args.bbox_size, args.time_scale)),
                              torch.randn(size=(args.batch_size, args.vel_size, args.time_scale)),
                              torch.tensor([[0] * args.batch_size])]]
            # Instantiate Model objects
            preds, labels = self.model.test(
                test_data=test_data,
                check_file=self.checkpoint_filepath,
                args=args
            )
            # cal result
            pred_cpu = torch.Tensor.cpu(preds)
            label_cpu = torch.Tensor.cpu(labels)

            acc = accuracy_score(label_cpu, np.round(pred_cpu))
            f1 = f1_score(label_cpu, np.round(pred_cpu))
            pre_s = precision_score(label_cpu, np.round(pred_cpu))
            recall_s = recall_score(label_cpu, np.round(pred_cpu))
            auc = roc_auc_score(label_cpu, np.round(pred_cpu))
            contrix = confusion_matrix(label_cpu, np.round(pred_cpu))

            # batch_size d_model drop_rate num_layers  lr
            file_name = 'results/res_' + args.save_name + '.txt'
            with open(file_name, 'a') as f:
                f.write(
                    f'lr: {args.lr} d_model: {args.d_model} batch_size: {args.batch_size} num_layers: {args.num_layers} drop_rate: {args.drop_rate}\n')
                # f.write(
                #     f'time_scale: {args.time_scale}\n time_crop: {args.time_crop}\n time_crop_scale: {args.time_crop_scale}\n')
                f.write(
                    f'Acc: {acc}\n f1: {f1}\n precision_score: {pre_s}\n recall_score: {recall_s}\n roc_auc_score: {auc}\n confusion_matrix: {contrix}\n\n')

            print(
                f'Acc: {acc}\n f1: {f1}\n precision_score: {pre_s}\n recall_score: {recall_s}\n roc_auc_score: {auc}\n confusion_matrix: {contrix}')

    def seed_all(self, seed):
        torch.cuda.empty_cache()
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
