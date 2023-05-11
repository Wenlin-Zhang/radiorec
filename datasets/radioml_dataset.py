import os.path
import h5py
import numpy as np
import pandas as pd
from typing import Optional, Callable
import torch


class RadioML2016(torch.utils.data.Dataset):
    """RadioML Dataset Example using RML2016.10a

    Args:
        root (:obj:`string`):
            Root directory where 'RML2016.10a_dict.pkl' exists. File can be downloaded from https://www.deepsig.ai/datasets

        transform (callable, optional):
            A function/transform that takes in an IQ vector and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None
    ):
        super(RadioML2016, self).__init__()
        self.file_path = root + "RML2016.10a_dict.pkl"
        data = pd.read_pickle(self.file_path)
        snrs = []
        mods = []
        iq_data = []
        for k in data.keys():
            for idx in range(len(data[k])):
                mods.append(k[0])
                snrs.append(k[1])
                iq_data.append(
                    np.asarray(data[k][idx][::2] + 1j * data[k][idx][1::2]).squeeze()
                )
        data_dict = {"class_name": mods, "snr": snrs, "data": iq_data}
        self.data_table = pd.DataFrame(data_dict)
        classes = list(self.data_table.class_name.unique())
        self.class_dict = dict(zip(classes, range(len(classes))))
        self.transform = transform

    def __getitem__(self, item: int):
        x = self.data_table["data"].iloc[item]
        if self.transform:
            x = self.transform(x)

        class_name = self.data_table["class_name"].iloc[item]
        y = self.class_dict[class_name]

        return x, y

    def __len__(self) -> int:
        return self.data_table.shape[0]


####
# 数据集包含24种调制样式：'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK',
# '32PSK', '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
# '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
# 'FM', 'GMSK', 'OQPSK'
# 信噪比范围：SNR = 【-20:32:2】 -20到30间隔2，共26种信噪比，具体【-20，-18......28,30】
# 每个调制，每个信噪比有4096条数据，共计2555904（24 * 26* 4096）条数据，IQ数据，数据格式为（1024, 2）
#
# 在数据集存放文件hdf5文件，存在3个参数X(IQ信号)，Y(调制方式），Z（信噪比）
# X——>[2555904，1024，2] Y——>[2555904，24] 采用onehot，所以不是[24264096，1] ，
# 顺序为上述顺序（class-fixed.txt修正后的 ） Z——>[2555904，1]
class RadioML2018(torch.utils.data.Dataset):
    """RadioML Dataset Example using RML2018.01

    Args:
        root (:obj:`string`):
            Root directory where 'GOLD_XYZ_OSC.0001_1024.hdf5' exists. File can be downloaded from https://www.deepsig.ai/datasets

        transform (callable, optional):
            A function/transform that takes in an IQ vector and returns a transformed version.

    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        min_db: int = 25,  # min SNR: 0 ==> -20db
        max_db: int = 26,  # max SNR: 26 ==> 30db
        complex: bool = True  # if true, return complex value, else return i/q as two channel
    ):
        super(RadioML2018, self).__init__()
        path = os.path.join(root, "GOLD_XYZ_OSC.0001_1024.hdf5")

        # Open the dataset
        hdf5_file = h5py.File(path, "r")

        # Read the HDF5 groups
        min_index = min_db * 4096
        max_index = max_db * 4096
        self.data = hdf5_file["X"][min_index:max_index]
        self.modulation_onehot = hdf5_file["Y"][min_index:max_index]
        self.snr = hdf5_file["Z"][min_index:max_index]
        for i in range(1, 24, 1):
            self.data = np.vstack((self.data, hdf5_file['X'][4096 * 26 * i + 15 * 4096:4096 * 26 * i + 16 * 4096]))
            self.modulation_onehot = np.vstack((self.modulation_onehot, hdf5_file['Y'][4096 * 26 * i + 15 * 4096:4096 * 26 * i + 16 * 4096]))
            self.snr = np.vstack((self.snr, hdf5_file['Z'][4096 * 26 * i + 15 * 4096:4096 * 26 * i + 16 * 4096]))

        # Class list corrected from `classes.txt` file
        self.class_list = [
            "OOK",
            "4ASK",
            "8ASK",
            "BPSK",
            "QPSK",
            "8PSK",
            "16PSK",
            "32PSK",
            "16APSK",
            "32APSK",
            "64APSK",
            "128APSK",
            "16QAM",
            "32QAM",
            "64QAM",
            "128QAM",
            "256QAM",
            "AM-SSB-WC",
            "AM-SSB-SC",
            "AM-DSB-WC",
            "AM-DSB-SC",
            "FM",
            "GMSK",
            "OQPSK",
        ]
        self.transform = transform
        self.complex = complex

    def __getitem__(self, item: int):
        x = self.data[item]
        if self.transform:
            x = self.transform(x)

        y = np.argmax(self.modulation_onehot[item])

        return x, y

    def __len__(self) -> int:
        return self.data.shape[0]
