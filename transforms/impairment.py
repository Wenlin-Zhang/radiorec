import numpy as np
from scipy import signal as sp
from typing import Optional, Any, Union, List, Literal, Optional
from transforms.transforms import SignalTransform
from transforms.functional import NumericParameter, FloatParameter, IntParameter
from transforms.functional import to_distribution, uniform_continuous_distribution, uniform_discrete_distribution
import transforms.functional as F

class Normalize(SignalTransform):
    """Normalize a IQ vector with mean and standard deviation.

    Args:
        norm :obj:`string`:
            Type of norm with which to normalize

        flatten :obj:`flatten`:
            Specifies if the norm should be calculated on the flattened
            representation of the input tensor

    Example:
        >>> import torchsig.transforms as ST
        >>> transform = ST.Normalize(norm=2) # normalize by l2 norm
        >>> transform = ST.Normalize(norm=1) # normalize by l1 norm
        >>> transform = ST.Normalize(norm=2, flatten=True) # normalize by l1 norm of the 1D representation

    """

    def __init__(
        self,
        norm: Optional[Union[int, float, Literal["fro", "nuc"]]] = 2,
        flatten: bool = False,
    ) -> None:
        super(Normalize, self).__init__()
        self.norm = norm
        self.flatten = flatten
        self.string: str = (
            self.__class__.__name__
            + "("
            + "norm={}, ".format(norm)
            + "flatten={}".format(flatten)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: np.ndarray) -> np.ndarray:
        data = F.normalize(data, self.norm, self.flatten)
        return data


def freq_shift(tensor: np.ndarray, f_shift: float) -> np.ndarray:
    """Shifts each tensor in freq by freq_shift along the time dimension

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor to be frequency-shifted.

        f_shift (:obj:`float` or :class:`numpy.ndarray`):
            Frequency shift relative to the sample rate in range [-.5, .5]

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has been frequency shifted along time dimension of size tensor.shape
    """
    sinusoid = np.exp(2j * np.pi * f_shift * np.arange(tensor.shape[0], dtype=np.float64))
    return np.multiply(tensor, np.asarray(sinusoid))


def freq_shift_avoid_aliasing(tensor: np.ndarray, f_shift: float) -> np.ndarray:
    """Similar to `freq_shift` function but performs the frequency shifting at
    a higher sample rate with filtering to avoid aliasing

    Args:
        tensor (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor to be frequency-shifted.

        f_shift (:obj:`float` or :class:`numpy.ndarray`):
            Frequency shift relative to the sample rate in range [-.5, .5]

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has been frequency shifted along time dimension of size tensor.shape
    """
    # Match output size to input
    num_iq_samples = tensor.shape[0]

    # Interpolate up to avoid frequency wrap around during shift
    up = 2
    down = 1
    tensor = sp.resample_poly(tensor, up, down)

    # Filter around center to remove original alias effects
    num_taps = int(2 * np.ceil(50 * 2 * np.pi / (1 / up) / .125 / 22))  # fred harris rule of thumb * 2
    taps = sp.firwin(
        num_taps,
        (1 / up),
        width=(1 / up) * .02,
        window=sp.get_window("blackman", num_taps),
        scale=True
    )
    tensor = sp.fftconvolve(tensor, taps, mode="same")

    # Freq shift to desired center freq
    time_vector = np.arange(tensor.shape[0], dtype=np.float)
    tensor = tensor * np.exp(2j * np.pi * f_shift / up * time_vector)

    # Filter to remove out-of-band regions
    num_taps = int(2 * np.ceil(50 * 2 * np.pi / (1 / up) / .125 / 22))  # fred harris rule-of-thumb * 2
    taps = sp.firwin(
        num_taps,
        1 / up,
        width=(1 / up) * .02,
        window=sp.get_window("blackman", num_taps),
        scale=True
    )
    tensor = sp.fftconvolve(tensor, taps, mode="same")
    tensor = tensor[:int(num_iq_samples * up)]  # prune to be correct size out of filter

    # Decimate back down to correct sample rate
    tensor = sp.resample_poly(tensor, down, up)

    return tensor[:num_iq_samples]


# 随机频偏变换
# 数据必须是复数
class RandomFrequencyShift(SignalTransform):
    """Shifts each tensor in freq by freq_shift along the time dimension.

    Args:
        freq_shift (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling freq_shift()
            * If int or float, freq_shift is fixed at the value provided
            * If list, freq_shift is any element in the list
            * If tuple, freq_shift is in range of (tuple[0], tuple[1])

    Example:
        >>> # Frequency shift inputs with uniform distribution in -fs/4 and fs/4
        >>> transform = RandomFrequencyShift(lambda size: np.random.uniform(-.25, .25, size))
        >>> # Frequency shift inputs always fs/10
        >>> transform = RandomFrequencyShift(lambda size: np.random.choice([.1], size))
        >>> # Frequency shift inputs with normal distribution with stdev .1
        >>> transform = RandomFrequencyShift(lambda size: np.random.normal(0, .1, size))
        >>> # Frequency shift inputs with uniform distribution in -fs/4 and fs/4
        >>> transform = RandomFrequencyShift((-.25, .25))
        >>> # Frequency shift all inputs by fs/10
        >>> transform = RandomFrequencyShift(.1)
        >>> # Frequency shift inputs with either -fs/4 or fs/4 (discrete)
        >>> transform = RandomFrequencyShift([-.25, .25])

    """

    def __init__(
            self,
            freq_shift: NumericParameter = uniform_continuous_distribution(-.5, .5),
            avoid_aliasing: bool = False
    ):
        super(RandomFrequencyShift, self).__init__()
        self.freq_shift = to_distribution(freq_shift, self.random_generator)
        self.avoid_aliasing = avoid_aliasing

    def __call__(self, data: Any) -> Any:
        freq_shift = self.freq_shift()
        if self.avoid_aliasing:
            # If any potential aliasing detected, perform shifting at higher sample rate
            new_data = freq_shift_avoid_aliasing(data, freq_shift)
        else:
            # Otherwise, use faster freq shifter
            new_data = freq_shift(data, freq_shift)
        return new_data


def phase_offset(tensor: np.ndarray, phase: float) -> np.ndarray:
    """ Applies a phase rotation to tensor

    Args:
        tensor: (:class:`numpy.ndarray`):
            (batch_size, vector_length, ...)-sized tensor.

        phase (:obj:`float`):
            phase to rotate sample in [-pi, pi]

    Returns:
        transformed (:class:`numpy.ndarray`):
            Tensor that has undergone a phase rotation

    """
    return tensor*np.exp(1j*phase)


# 随机相位变化
class RandomPhaseShift(SignalTransform):
    """Applies a random phase offset to tensor

    Args:
        phase_offset (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            * If Callable, produces a sample by calling phase_offset()
            * If int or float, phase_offset is fixed at the value provided
            * If list, phase_offset is any element in the list
            * If tuple, phase_offset is in range of (tuple[0], tuple[1])

    Example:
        >>> # Phase Offset in range [-pi, pi]
        >>> transform = RandomPhaseShift(uniform_continuous_distribution(-1, 1))
        >>> # Phase Offset from [-pi/2, 0, and pi/2]
        >>> transform = RandomPhaseShift(uniform_discrete_distribution([-.5, 0, .5]))
        >>> # Phase Offset in range [-pi, pi]
        >>> transform = RandomPhaseShift((-1, 1))
        >>> # Phase Offset either -pi/4 or pi/4
        >>> transform = RandomPhaseShift([-.25, .25])
        >>> # Phase Offset is fixed at -pi/2
        >>> transform = RandomPhaseShift(-.5)
    """
    def __init__(
        self,
        phase_offset: FloatParameter = uniform_continuous_distribution(-1, 1),
        **kwargs
    ):
        super(RandomPhaseShift, self).__init__(**kwargs)
        self.phase_offset = to_distribution(phase_offset, self.random_generator)

    def __call__(self, data: Any) -> Any:
        phases = self.phase_offset()
        data = phase_offset(data, phases*np.pi)
        return data

class SpectrogramPatchShuffle(SignalTransform):
    """Randomly shuffle multiple local regions of samples.

    Transform is loosely based on
    `PatchShuffle Regularization <https://arxiv.org/pdf/1707.07103.pdf>`_.

    Args:
         patch_size (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            patch_size sets the size of each patch to shuffle
            * If Callable, produces a sample by calling patch_size()
            * If int or float, patch_size is fixed at the value provided
            * If list, patch_size is any element in the list
            * If tuple, patch_size is in range of (tuple[0], tuple[1])

        shuffle_ratio (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            shuffle_ratio sets the ratio of the patches to shuffle
            * If Callable, produces a sample by calling shuffle_ratio()
            * If int or float, shuffle_ratio is fixed at the value provided
            * If list, shuffle_ratio is any element in the list
            * If tuple, shuffle_ratio is in range of (tuple[0], tuple[1])

    """

    def __init__(
        self,
        patch_size: NumericParameter = (2, 16),
        shuffle_ratio: FloatParameter = (0.01, 0.10),
    ) -> None:
        super(SpectrogramPatchShuffle, self).__init__()
        self.patch_size = to_distribution(patch_size, self.random_generator)
        self.shuffle_ratio = to_distribution(shuffle_ratio, self.random_generator)
        self.string = (
            self.__class__.__name__
            + "("
            + "patch_size={}, ".format(patch_size)
            + "shuffle_ratio={}".format(shuffle_ratio)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        patch_size = int(self.patch_size())
        shuffle_ratio = self.shuffle_ratio()

        output: np.ndarray = F.spec_patch_shuffle(
            data,
            patch_size,
            shuffle_ratio,
        )
        return output

class AddNoise(SignalTransform):
    """Add random AWGN at specified power levels

    Note:
        Differs from the TargetSNR() in that this transform adds
        noise at a specified power level, whereas TargetSNR()
        assumes a basebanded signal and adds noise to achieve a specified SNR
        level for the signal of interest. This transform,
        AddNoise() is useful for simply adding a randomized
        level of noise to either a narrowband or wideband input.

    Args:
        noise_power_db (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            Defined as 10*log10(np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2)) if in dB,
            np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2) if linear.

            * If Callable, produces a sample by calling target_snr()
            * If int or float, target_snr is fixed at the value provided
            * If list, target_snr is any element in the list
            * If tuple, target_snr is in range of (tuple[0], tuple[1])

        input_noise_floor_db (:obj:`float`):
            The noise floor of the input data in dB

        linear (:obj:`bool`):
            If True, target_snr and signal_power is on linear scale not dB.

    Example:
        >>> import torchsig.transforms as ST
        >>> # Added AWGN power range is (-40, -20) dB
        >>> transform = ST.AddNoise((-40, -20))

    """

    def __init__(
        self,
        noise_power_db: NumericParameter = uniform_continuous_distribution(-80, -60),
        input_noise_floor_db: float = 0.0,
        linear: bool = False,
        **kwargs,
    ) -> None:
        super(AddNoise, self).__init__(**kwargs)
        self.noise_power_db = to_distribution(noise_power_db, self.random_generator)
        self.input_noise_floor_db = input_noise_floor_db
        self.linear = linear
        self.string = (
            self.__class__.__name__
            + "("
            + "noise_power_db={}, ".format(noise_power_db)
            + "input_noise_floor_db={}, ".format(input_noise_floor_db)
            + "linear={}".format(linear)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        noise_power_db = self.noise_power_db(size=data.shape[0])
        noise_power_db = 10 * np.log10(noise_power_db) if self.linear else noise_power_db
        output: np.ndarray = F.awgn(data, noise_power_db)
        return output


class TimeVaryingNoise(SignalTransform):
    """Add time-varying random AWGN at specified input parameters

    Args:
        noise_power_db_low (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            Defined as 10*log10(np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2)) if in dB,
            np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2) if linear.
            * If Callable, produces a sample by calling noise_power_db_low()
            * If int or float, noise_power_db_low is fixed at the value provided
            * If list, noise_power_db_low is any element in the list
            * If tuple, noise_power_db_low is in range of (tuple[0], tuple[1])

        noise_power_db_high (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            Defined as 10*log10(np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2)) if in dB,
            np.mean(np.abs(x)**2)/np.mean(np.abs(n)**2) if linear.
            * If Callable, produces a sample by calling noise_power_db_low()
            * If int or float, noise_power_db_low is fixed at the value provided
            * If list, noise_power_db_low is any element in the list
            * If tuple, noise_power_db_low is in range of (tuple[0], tuple[1])

        inflections (:py:class:`~Callable`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`):
            Number of inflection points in time-varying noise
            * If Callable, produces a sample by calling inflections()
            * If int or float, inflections is fixed at the value provided
            * If list, inflections is any element in the list
            * If tuple, inflections is in range of (tuple[0], tuple[1])

        random_regions (:py:class:`~Callable`, :obj:`bool`, :obj:`list`, :obj:`tuple`):
            If inflections > 0, random_regions specifies whether each
            inflection point should be randomly selected or evenly divided
            among input data
            * If Callable, produces a sample by calling random_regions()
            * If bool, random_regions is fixed at the value provided
            * If list, random_regions is any element in the list

        linear (:obj:`bool`):
            If True, powers input are on linear scale not dB.

    """

    def __init__(
        self,
        noise_power_db_low: NumericParameter = uniform_continuous_distribution(-80, -60),
        noise_power_db_high: NumericParameter = uniform_continuous_distribution(-40, -20),
        inflections: IntParameter = uniform_continuous_distribution(0, 10),
        random_regions: Union[List, bool] = True,
        linear: bool = False,
        **kwargs,
    ) -> None:
        super(TimeVaryingNoise, self).__init__(**kwargs)
        self.noise_power_db_low = to_distribution(noise_power_db_low)
        self.noise_power_db_high = to_distribution(noise_power_db_high)
        self.inflections = to_distribution(inflections)
        self.random_regions = to_distribution(random_regions)
        self.linear = linear
        self.string = (
            self.__class__.__name__
            + "("
            + "noise_power_db_low={}, ".format(noise_power_db_low)
            + "noise_power_db_high={}, ".format(noise_power_db_high)
            + "inflections={}, ".format(inflections)
            + "random_regions={}, ".format(random_regions)
            + "linear={}".format(linear)
            + ")"
        )

    def __repr__(self) -> str:
        return self.string

    def __call__(self, data: Any) -> Any:
        noise_power_db_low = self.noise_power_db_low()
        noise_power_db_high = self.noise_power_db_high()
        noise_power_db_low = (
            10 * np.log10(noise_power_db_low) if self.linear else noise_power_db_low
        )
        noise_power_db_high = (
            10 * np.log10(noise_power_db_high) if self.linear else noise_power_db_high
        )
        inflections = int(self.inflections())
        random_regions = self.random_regions()

        output: np.ndarray = F.time_varying_awgn(
            data,
            noise_power_db_low,
            noise_power_db_high,
            inflections,
            random_regions,
        )
        return output