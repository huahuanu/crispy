# basic package
import os
from pathlib import Path
import torch.utils.data as data
from tqdm import tqdm
import torch
import pickle5 as P
import numpy as np
# local package
from data.jaad_data import JAAD as SET


class JaadDataset(data.Dataset):
    def __init__(self, data_set, data_path, set_path, balance=False, transforms=None):
        super(JaadDataset, self).__init__()

        self.transforms = transforms
        self.data_set = data_set
        self.balance = balance
        self.maxw_var = 9
        self.maxh_var = 6
        self.maxd_var = 2

        path = data_path

        self.input_size = int(32 * 1)
        if data_set == 'train':
            num_samples = [1025, 4778, 17582]
        elif data_set == 'val':
            num_samples = [176, 454, 2772]
        elif data_set == 'test':
            num_samples = [1871, 3204, 13037]

        balance_data = [max(num_samples) / s for s in num_samples]

        self.data_path = os.getcwd() / Path(path) / 'data'
        self.imgs_path = os.getcwd() / Path(path) / 'imgs'
        self.data_list = [data_name for data_name in os.listdir(self.data_path)]
        self.set_path = set_path

        imdb = SET(data_path=self.set_path)
        self.vid_ids = imdb._get_video_ids_split(data_set)

        filt_list = lambda x: '_'.join(x.split('_')[:2]) in self.vid_ids
        ped_ids = list(filter(filt_list, self.data_list))

        self.ped_data = {}
        # ped_ids = ped_ids[:1000]

        for ped_id in tqdm(ped_ids, desc=f'loading {data_set} data in memory'):
            ped_path = self.data_path.joinpath(ped_id).as_posix()
            loaded_data = self.load_data(ped_path)
            img_file = self.imgs_path.joinpath(loaded_data['crop_img'].stem + '.pkl').as_posix()
            loaded_data['crop_img'] = self.load_data(img_file)

            if balance:
                if 'b' not in ped_id:  # irrelevant
                    self.repet_data(balance_data[2], loaded_data, ped_id)
                elif loaded_data['crossing'] == 0:  # not cross
                    self.repet_data(balance_data[0], loaded_data, ped_id)
                elif loaded_data['crossing'] == 1:  # crossing
                    self.repet_data(balance_data[1], loaded_data, ped_id)
            else:
                self.ped_data[ped_id.split('.')[0]] = loaded_data

        self.ped_ids = list(self.ped_data.keys())
        self.data_len = len(self.ped_ids)

    def load_data(self, ped_path):
        with open(ped_path, 'rb') as f:
            data = P.load(f, encoding='bytes')
        return data

    def repet_data(self, num, data, ped_id):
        ped_id = ped_id.split('.')[0]
        if self.data_set == 'train' or self.data_set == 'val':
            prov = num % 1
            num = int(num) if prov == 0 else int(num) + np.random.choice(2, 1, p=[1 - prov, prov])[0]
        else:
            num = int(num)

        for i in range(int(num)):
            self.ped_data[ped_id + f'-r{i}'] = data

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):

        ped_id = self.ped_ids[item]
        ped_data = self.ped_data[ped_id]
        w, h = ped_data['w'], ped_data['h']

        # # keypoints  -> size: [4, 30, 19]:[C, T, N]
        # kp = ped_data['kps']
        # if self.data_set == 'train':
        #     kp[..., 0] = np.clip(kp[..., 0] + np.random.randint(self.maxw_var, size=kp[..., 0].shape), 0, w)
        #     kp[..., 1] = np.clip(kp[..., 1] + np.random.randint(self.maxh_var, size=kp[..., 1].shape), 0, h)
        #     kp[..., 2] = np.clip(kp[..., 2] + np.random.randint(self.maxd_var, size=kp[..., 2].shape), 0, 80)
        # kp[..., 0] /= w
        # kp[..., 1] /= h
        # kp[..., 2] /= 80
        # kp = torch.from_numpy(kp.transpose(2, 0, 1)).float().contiguous()
        # kp = kp[:, :30, :] # not use future kps data.

        # # seg map and image  -> size: [4, 192, 64]:[C, H, W]
        # seg_map = torch.from_numpy(ped_data['crop_img'][:1]).float()
        # seg_map = (seg_map - 78.26) / 45.12
        # img = ped_data['crop_img'][1:]
        # img = self.transforms(img.transpose(1, 2, 0)).contiguous()
        # img = torch.cat([seg_map, img], dim=0)

        # velocity  -> size: [2, 30]:[C, T]
        vel = torch.from_numpy(np.tile(ped_data['obd_speed'], [1, 2]).transpose(1, 0)).float().contiguous()

        label = torch.from_numpy(np.asarray(ped_data['crossing'])).float()

        # # trajectory
        # traj = ped_data['obs_traj']
        # traj[:, 0] /= w
        # traj[:, 1] /= h
        # traj = torch.from_numpy(traj).float().contiguous()

        # traj_future = ped_data['future_traj']
        # traj_future[:, 0] /= w
        # traj_future[:, 1] /= h
        # traj_future = torch.from_numpy(traj_future).float().contiguous()

        # end_point
        # end_point = ped_data['endp_point'] if ped_data['endp_point'] is not None else ped_data['future_traj'][-1]
        # end_point = torch.from_numpy(end_point).float().contiguous()

        # bbox -> size: [4, 30]:[C, T]
        bbox = np.asarray(ped_data['bbox'])
        bbox[:, 0] /= w
        bbox[:, 1] /= h
        bbox[:, 2] /= w
        bbox[:, 3] /= h
        bbox = torch.from_numpy(bbox).float().contiguous().transpose(0, 1)

        return bbox, vel, label
