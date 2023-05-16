import numpy as np
from scipy import signal as sp
from typing import Optional, Any, Union, List
from transforms.transforms import SignalTransform
from transforms.functional import NumericParameter, FloatParameter
from transforms.functional import to_distribution, uniform_continuous_distribution, uniform_discrete_distribution


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
