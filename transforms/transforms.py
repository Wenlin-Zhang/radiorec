import warnings
import numpy as np
from copy import deepcopy
from typing import Any, List, Callable, Optional


class SignalTransform:
    """An abstract class representing a Transform that can either work on
    targets or data

    """

    def __init__(self):
        self.random_generator = np.random

    def __call__(self, data: Any) -> Any:
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Compose(SignalTransform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects):
            list of transforms to compose.

    Example:
        >>> import transforms as ST
        >>> transform = ST.Compose([ST.AddNoise(10), ST.InterleaveComplex()])

    """

    def __init__(self, transforms: List[SignalTransform], **kwargs):
        super(Compose, self).__init__(**kwargs)
        self.transforms = transforms

    def __call__(self, data: Any) -> Any:
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        return "\n".join([str(t) for t in self.transforms])


class NoTransform(SignalTransform):
    """Just passes the data -- surprisingly useful in pipelines

    Example:
        >>> import transforms as ST
        >>> transform = ST.NoTransform()

    """

    def __init__(self, **kwargs):
        super(NoTransform, self).__init__(**kwargs)

    def __call__(self, data: Any) -> Any:
        return data


class Lambda(SignalTransform):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.

    Example:
        >>> import transforms as ST
        >>> transform = ST.Lambda(lambda x: x**2)  # A transform that squares all inputs.

    """

    def __init__(self, func: Callable, **kwargs):
        super(Lambda, self).__init__(**kwargs)
        self.func = func

    def __call__(self, data: Any) -> Any:
        return self.func(data)


class FixedRandom(SignalTransform):
    """Restricts a randomized transform to apply only a fixed set of seeds.
    For example, this could be used to add noise randomly from among 1000
    possible sets of noise or add fading from 1000 possible channels.

    Args:
        transform (:obj:`Callable`):
            transform to be called

        num_seeds (:obj:`int`):
            number of possible random seeds to use

    Example:
        >>> import transforms as ST
        >>> transform = ST.FixedRandom(ST.AddNoise(10), num_seeds=10)

    """

    def __init__(self, transform: SignalTransform, num_seeds: int, **kwargs):
        super(FixedRandom, self).__init__(**kwargs)
        self.transform = transform
        self.num_seeds = num_seeds

    def __call__(self, data: Any) -> Any:
        seed = self.random_generator.choice(self.num_seeds)
        orig_state = (
            np.random.get_state()
        )  # we do not want to somehow fix other random number generation processes.
        np.random.seed(seed)
        data = self.transform(data)
        np.random.set_state(orig_state)  # return numpy back to its previous state
        return data


class RandomApply(SignalTransform):
    """Randomly applies a set of transforms with probability p

    Args:
        transform (``Transform`` objects):
            transform to randomly apply

        probability (:obj:`float`):
            In [0, 1.0], the probability with which to apply a transform

    Example:
        >>> import transforms as ST
        >>> transform = ST.RandomApply(ST.AddNoise(10), probability=.5)  # Add 10dB noise with probability .5

    """

    def __init__(self, transform: SignalTransform, probability: float, **kwargs):
        super(RandomApply, self).__init__(**kwargs)
        self.transform = transform
        self.probability = probability

    def __call__(self, data: Any) -> Any:
        return (
            self.transform(data)
            if self.random_generator.rand() < self.probability
            else data
        )

