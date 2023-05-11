import numpy
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable, Any
from scipy import signal
from transforms.transforms import SignalTransform


@dataclass
class SpectrogramConfig:
    nperseg : int = 256
    noverlap : int = 128
    nfft : int = 256
    window : str = 'hann'
    return_onesided : bool = False
    mode: str = 'magnitude'


def spectrogram(
            data: np.ndarray,  # one dimensional complex or float data array
            config: SpectrogramConfig
    ) -> np.ndarray:
    _, _, spectrograms = signal.spectrogram(
        data,
        nperseg = config.nperseg,
        noverlap = config.noverlap,
        nfft = config.nfft,
        window = config.window,
        return_onesided = config.return_onesided,
        mode = config.mode,
        axis = 0
    )
    image = np.fft.fftshift(spectrograms, axes = 0)
    if config.mode == 'complex':
        rimage = np.real(image)
        iimage = np.imag(image)
        image = np.stack([rimage, iimage])
    else:
        image = np.expand_dims(image, axis = 0) # expand to 1 channel
    return image


class Spectrogram(SignalTransform):

    def __init__(
            self,
            config: SpectrogramConfig
    ):
        super(Spectrogram, self).__init__()
        self.config = config

    def __call__(self, data: Any) -> Any:
        data = spectrogram(data, self.config)
        return data
