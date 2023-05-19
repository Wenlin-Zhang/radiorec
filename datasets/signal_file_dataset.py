import os
import os.path

import numpy
import pandas as pd
import numpy as np
from dataclasses import dataclass
from collections import Counter
from numpy.lib.stride_tricks import as_strided
import torch
from typing import Tuple, Optional, Callable, Any
import h5py


class SignalFileSet:
    def __init__(self, root_dir = None, suffix = None):
        if root_dir is None:
            self.root_dir = ''   # root directory
            self.class_list = []   # class array
            self.file_dict = {}  # file dictionary  (file_name, class_index)
            self.class_dict = {} # class dictionary (class, class_index)
            return

        self.root_dir = root_dir
        self.class_list = [subdir for subdir in os.listdir(root_dir)
                    if os.path.isdir(os.path.join(root_dir, subdir))]
        self.class_list.sort()
        self.class_dict = {label:index for index, label in enumerate(self.class_list)}

        self.file_dict = {}
        for index, label in enumerate(self.class_list):
            subdir = os.path.join(root_dir, label)

            files = [file for file in os.listdir(subdir)
                     if os.path.isfile(os.path.join(subdir, file)) and file.endswith(suffix)]
            for file in files:
                self.file_dict[os.path.join(label, file)] = index

    def write_csv(self, csv_file_path: str):
        file_list = list(self.file_dict.keys())
        label_list = list(self.file_dict.values())
        label_list = map(lambda index: self.class_list[index], label_list)
        df = pd.DataFrame({'file': file_list, 'label': label_list})
        df.to_csv(csv_file_path, header = True, index = None)

    def read_csv(self, csv_file_path: str, root_dir: str = None):
        if root_dir is None:
            self.root_dir = os.path.dirname(csv_file_path)
        else:
            self.root_dir = root_dir

        df = pd.read_csv(csv_file_path, index_col='file', dtype = {'label': str})
        self.class_list = df['label'].unique().tolist()
        self.class_list.sort()
        self.class_dict = {label:index for index, label in enumerate(self.class_list)}

        df['label'] = df['label'].map(self.class_dict)
        self.file_dict = df['label'].to_dict()

    def print_info(self):
        print(f'data path: {self.root_dir}')
        print(f'classes: {self.class_list}')
        print(f'class dict: {self.class_dict}')
        c = Counter(self.file_dict.values())
        label_count = {self.class_list[index]: count for index, count in c.items()}
        print(label_count)

@dataclass
class SignalFileConfig:
    fs : int = 50000     # sample rate
    seg_len: int = 20  # segment length in ms
    seg_shift: int = 10     # segment shift in ms (default = seg_len, no overlap)
    max_num: int = 1000   # max segments per file
    is_complex: bool = True  # if IQ data
    energy_threshold: float = 10   # energy threshold, < 0: no filter


def filter_with_energy(x, y=None, energy_threshold=6.0):  # x: (cnt, seg_len, 2)
    if energy_threshold <= 0.0:
        return x, y
    magnitude = np.linalg.norm(x, axis = -1)
    energy = np.linalg.norm(magnitude, axis = -1)
    mask = (energy > energy_threshold)
    xx = x[mask]
    if y:
        yy = y[mask]
    else:
        yy = None
    return xx, yy


def get_segment_data(filepath:str, seg_len, seg_shift, max_num, step = 2):
    data = np.fromfile(filepath, dtype="float32")
    file_len = len(data)
    if file_len < seg_len * step:
        return None
    if seg_shift == seg_len:
        count = int(min(file_len // (step * seg_len), max_num))
        data = data[:count * seg_len * step]
        data = data.reshape((count, seg_len, step))
    else:
        count = int(min((file_len - seg_len * step) // (step * seg_shift) + 1, max_num))
        data = data[: ((count - 1) * seg_shift + seg_len) * step]
        data = as_strided(data,
                          shape=(count, seg_len * step),
                          strides=(data.itemsize * seg_shift * step, data.itemsize))
        data = data.reshape((count, seg_len, step))
    return data


def generate_dataset(file_set: SignalFileSet, conf: SignalFileConfig):
    seg_data_len = int(conf.fs * conf.seg_len // 1000)
    seg_shift_len = int(conf.fs * conf.seg_shift // 1000)
    step = 2 if conf.is_complex else 1
    dataset = []
    labels = []
    for file_name, label in file_set.file_dict.items():
        data = get_segment_data(os.path.join(file_set.root_dir, file_name), seg_len= seg_data_len, seg_shift=seg_shift_len,
                        max_num=conf.max_num, step=step)
        if data is None:
            print(f'Warning: file length less than a segment [{file_name}].')
            continue
        if conf.energy_threshold > 0:
            data, _ = filter_with_energy(data, energy_threshold=conf.energy_threshold)
            if len(data) == 0:
                print(f'Warning: no segment reach the energy threshold ({conf.energy_threshold}) [{file_name}].')
                continue
        dataset.append(data)
        labels.extend([label] * len(data))
    data = np.vstack(dataset)  # (total_count, seg_length, channel = 2)
    labels = np.array(labels)
    return data, labels


class SignalDataSet(torch.utils.data.Dataset):
    def __init__(
        self,
        fileset: Optional[SignalFileSet] = None,
        config: Optional[SignalFileConfig] = None,
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ):
        super(SignalDataSet, self).__init__()
        if fileset == None:
            self.fileset = None
            self.fs = 1
            self.data = None
            self.labels = None
            self.transform = None
            self.random_generator = np.random.RandomState()
            return
        self.transform = transform
        self.random_generator = np.random.RandomState(seed)
        self.class_list = fileset.class_list   # need save
        self.class_dict = fileset.class_dict   # need save
        self.fs = config.fs
        self.data, self.labels = generate_dataset(fileset, config)  # need save

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        x = self.data[index]
        y = self.labels[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.data)

    def print_info(self):
        print(f'class list: {self.class_list}')
        print(f'class dict: {self.class_dict}')
        c = Counter(self.labels)
        seg_count = {self.class_list[index]: count for index, count in c.items()}
        print(f'seg count for each class: {seg_count}')

    def save(self, fname: str):
        hdf5_file = h5py.File(fname, "w")
        hdf5_file['class_list'] = self.class_list
        hdf5_file['fs'] = [self.fs]
        hdf5_file['data'] = self.data
        hdf5_file['labels'] = self.labels
        hdf5_file.close()

    def load(self, fname: str):
        hdf5_file = h5py.File(fname, "r")
        self.class_list = [name.decode('utf-8') for name in hdf5_file['class_list']]
        self.class_dict = {label:index for index, label in enumerate(self.class_list)}
        self.fs = hdf5_file['fs'][0]
        self.data = numpy.array(hdf5_file['data'])
        self.labels = numpy.array(hdf5_file['labels'])
        hdf5_file.close()


def test():
    sigFileSet = SignalFileSet('test', 'bin')
    #sigFileSet.print_info()

    sigFileSet.write_csv('test/index.csv')

    sigFileSet2 = SignalFileSet()
    sigFileSet2.read_csv('test/index.csv')
    #sigFileSet2.print_info()

    conf = SignalFileConfig()
    dataset = SignalDataSet(sigFileSet2, conf)
    dataset.print_info()

if __name__ == "__main__":
    test()