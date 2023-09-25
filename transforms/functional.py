from typing import Callable, Union, Tuple, Literal, Optional, List
from functools import partial
import numpy as np
from scipy import signal as sp

FloatParameter = Union[Callable[[int], float], float, Tuple[float, float], List]
IntParameter = Union[Callable[[int], int], int, Tuple[int, int], List]
NumericParameter = Union[FloatParameter, IntParameter]


def uniform_discrete_distribution(choices: List, random_generator: np.random.RandomState = np.random.RandomState()):
    return partial(random_generator.choice, choices)


def uniform_continuous_distribution(
        lower: Union[int, float],
        upper: Union[int, float],
        random_generator: np.random.RandomState = np.random.RandomState()
):
    return partial(random_generator.uniform, lower, upper)


def to_distribution(param, random_generator: np.random.RandomState = np.random.RandomState()):
    if isinstance(param, Callable):
        return param

    if isinstance(param, list):
        if isinstance(param[0], tuple):
            tuple_from_list = param[random_generator.randint(len(param))]
            return uniform_continuous_distribution(
                tuple_from_list[0], 
                tuple_from_list[1], 
                random_generator,
            )
        return uniform_discrete_distribution(param, random_generator)

    if isinstance(param, tuple):
        return uniform_continuous_distribution(param[0], param[1], random_generator)

    if isinstance(param, int) or isinstance(param, float):
        return uniform_discrete_distribution([param], random_generator)

    return param

def normalize(
    tensor: np.ndarray,
    norm_order: Optional[Union[float, int, Literal["fro", "nuc"]]] = 2,
    flatten: bool = False,
) -> np.ndarray:
    """Scale a tensor so that a specfied norm computes to 1. For detailed information, see :func:`numpy.linalg.norm.`
        * For norm=1,      norm = max(sum(abs(x), axis=0)) (sum of the elements)
        * for norm=2,      norm = sqrt(sum(abs(x)^2), axis=0) (square-root of the sum of squares)
        * for norm=np.inf, norm = max(sum(abs(x), axis=1)) (largest absolute value)

    Args:
        tensor (:class:`numpy.ndarray`)):
            (batch_size, vector_length, ...)-sized tensor to be normalized.

        norm_order (:class:`int`)):
            norm order to be passed to np.linalg.norm

        flatten (:class:`bool`)):
            boolean specifying if the input array's norm should be calculated on the flattened representation of the input tensor

    Returns:
        Tensor:
            Normalized complex array.
    """
    if flatten:
        flat_tensor = tensor.reshape(tensor.size)
        norm = np.linalg.norm(flat_tensor, norm_order, keepdims=True)
    else:
        norm = np.linalg.norm(tensor, norm_order, keepdims=True)
    return np.multiply(tensor, 1.0 / norm)

def resample(
    tensor: np.ndarray,
    up_rate: int,
    down_rate: int,
    num_iq_samples: int,
    keep_samples: bool,
    anti_alias_lpf: bool = False,
) -> np.ndarray:
    """Resample a tensor by rational value

    Args:
        tensor (:class:`numpy.ndarray`):
            tensor to be resampled.

        up_rate (:class:`int`):
            rate at which to up-sample the tensor

        down_rate (:class:`int`):
            rate at which to down-sample the tensor

        num_iq_samples (:class:`int`):
            number of IQ samples to have after resampling

        keep_samples (:class:`bool`):
            boolean to specify if the resampled data should be returned as is

        anti_alias_lpf (:class:`bool`)):
            boolean to specify if an additional anti aliasing filter should be
            applied

    Returns:
        Tensor:
            Resampled tensor
    """
    if anti_alias_lpf:
        new_rate = up_rate / down_rate
        taps = low_pass(
            cutoff=new_rate * 0.98 / 2,
            transition_bandwidth=(0.5 - (new_rate * 0.98) / 2) / 4,
        )
        tensor = sp.convolve(tensor, taps, mode="same")

    # Resample
    resampled = sp.resample_poly(tensor, up_rate, down_rate)

    # Handle extra or not enough IQ samples
    if keep_samples:
        new_tensor = resampled
    elif resampled.shape[0] > num_iq_samples:
        new_tensor = resampled[-num_iq_samples:]
    else:
        new_tensor = np.zeros((num_iq_samples,), dtype=np.complex128)
        new_tensor[: resampled.shape[0]] = resampled

    return new_tensor

def awgn(tensor: np.ndarray, noise_power_db: float) -> np.ndarray:
    """Adds zero-mean complex additive white Gaussian noise with power of
    noise_power_db.

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        noise_power_db (:obj:`float`):
            Defined as 10*log10(E[|n|^2]).

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor with added noise.
    """
    real_noise = np.random.randn(*tensor.shape)
    imag_noise = np.random.randn(*tensor.shape)
    return tensor + (10.0 ** (noise_power_db / 20.0)) * (real_noise + 1j * imag_noise) / np.sqrt(2)


def time_varying_awgn(
    tensor: np.ndarray,
    noise_power_db_low: float,
    noise_power_db_high: float,
    inflections: int,
    random_regions: bool,
) -> np.ndarray:
    """Adds time-varying complex additive white Gaussian noise with power
    levels in range (`noise_power_db_low`, `noise_power_db_high`) and with
    `inflections` number of inflection points spread over the input tensor
    randomly if `random_regions` is True or evely spread if False

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        noise_power_db_low (:obj:`float`):
            Defined as 10*log10(E[|n|^2]).

        noise_power_db_high (:obj:`float`):
            Defined as 10*log10(E[|n|^2]).

        inflections (:obj:`int`):
            Number of inflection points for time-varying nature

        random_regions (:obj:`bool`):
            Specify if inflection points are randomly spread throughout tensor
            or if evenly spread

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor with added noise.
    """
    real_noise: np.ndarray = np.random.randn(*tensor.shape)
    imag_noise: np.ndarray = np.random.randn(*tensor.shape)
    noise_power_db: np.ndarray = np.zeros(tensor.shape)

    if inflections == 0:
        inflection_indices = np.array([0, tensor.shape[0]])
    else:
        if random_regions:
            inflection_indices = np.sort(
                np.random.choice(tensor.shape[0], size=inflections, replace=False)
            )
            inflection_indices = np.append(inflection_indices, tensor.shape[0])
            inflection_indices = np.insert(inflection_indices, 0, 0)
        else:
            inflection_indices = np.arange(inflections + 2) * int(
                tensor.shape[0] / (inflections + 1)
            )

    for idx in range(len(inflection_indices) - 1):
        start_idx = inflection_indices[idx]
        stop_idx = inflection_indices[idx + 1]
        duration = stop_idx - start_idx
        start_power = noise_power_db_low if idx % 2 == 0 else noise_power_db_high
        stop_power = noise_power_db_high if idx % 2 == 0 else noise_power_db_low
        noise_power_db[start_idx:stop_idx] = np.linspace(start_power, stop_power, duration)

    return tensor + (10.0 ** (noise_power_db / 20.0)) * (real_noise + 1j * imag_noise) / np.sqrt(2)

def patch_shuffle(
    tensor: np.ndarray,
    patch_size: int,
    shuffle_ratio: float,
) -> np.ndarray:
    """Apply shuffling of patches specified by `num_patches`

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        patch_size (:obj:`int`):
            Size of each patch to shuffle

        shuffle_ratio (:obj:`float`):
            Ratio of patches to shuffle

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone patch shuffling

    """
    num_patches = int(tensor.shape[0] / patch_size)
    num_to_shuffle = int(num_patches * shuffle_ratio)
    patches_to_shuffle = np.random.choice(
        num_patches,
        replace=False,
        size=num_to_shuffle,
    )

    for patch_idx in patches_to_shuffle:
        patch_start = int(patch_idx * patch_size)
        patch = tensor[patch_start : patch_start + patch_size]
        np.random.shuffle(patch)
        tensor[patch_start : patch_start + patch_size] = patch

    return tensor

def spec_patch_shuffle(
    tensor: np.ndarray,
    patch_size: int,
    shuffle_ratio: float,
) -> np.ndarray:
    """Apply shuffling of patches specified by `num_patches`

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        patch_size (:obj:`int`):
            Size of each patch to shuffle

        shuffle_ratio (:obj:`float`):
            Ratio of patches to shuffle

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone patch shuffling

    """
    channels, height, width = tensor.shape
    num_freq_patches = int(height / patch_size)
    num_time_patches = int(width / patch_size)
    num_patches = int(num_freq_patches * num_time_patches)
    num_to_shuffle = int(num_patches * shuffle_ratio)
    patches_to_shuffle = np.random.choice(
        num_patches,
        replace=False,
        size=num_to_shuffle,
    )

    for patch_idx in patches_to_shuffle:
        freq_idx = np.floor(patch_idx / num_time_patches)
        time_idx = patch_idx % num_time_patches
        patch = tensor[
            :,
            int(freq_idx * patch_size) : int(freq_idx * patch_size + patch_size),
            int(time_idx * patch_size) : int(time_idx * patch_size + patch_size),
        ]
        patch = patch.reshape(int(1 * patch_size * patch_size))
        np.random.shuffle(patch)
        patch = patch.reshape(1, int(patch_size), int(patch_size))
        tensor[
            :,
            int(freq_idx * patch_size) : int(freq_idx * patch_size + patch_size),
            int(time_idx * patch_size) : int(time_idx * patch_size + patch_size),
        ] = patch
    return tensor