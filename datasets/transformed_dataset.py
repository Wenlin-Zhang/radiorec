import torch
from typing import Tuple, Optional, Callable, Any


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            transform: Optional[Callable]
    ):
        super(TransformedDataset, self).__init__()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        x, y = self.dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.dataset)
