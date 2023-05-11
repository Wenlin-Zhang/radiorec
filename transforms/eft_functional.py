
import numpy as np


def interleave_complex(tensor: np.ndarray) -> np.ndarray:
    """Converts real interleaved IQ vector to complex vectors

    Args:
        tensor (:class:`numpy.ndarray`):
            (vector_length, 2) - sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            Interleaved vectors. (vector_length * 2)
    """
    new_tensor = tensor[:, 0] + 1j * tensor[:, 1]
    return new_tensor


def interleave_to_2d(tensor: np.ndarray) -> np.ndarray:
    """Converts interleaved IQ data to two channels representing real and imaginary

    Args:
        tensor (:class:`numpy.ndarray`):
            (vector_length, 2) - sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            Expanded vectors (2, vector_length)
    """

    new_tensor = np.transpose(tensor)
    return new_tensor


def real(tensor: np.ndarray) -> np.ndarray:
    """Converts interleaved IQ data to a real-only vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (vector_length, 2) - sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            real(tensor)
    """
    return tensor[:, 0]


def imag(tensor: np.ndarray) -> np.ndarray:
    """Converts interleaved IQ data to a imaginary-only vector

    Args:
        tensor (:class:`numpy.ndarray`):
            vector_length - sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            imag(tensor)
    """
    return tensor[:, 1]


def complex_magnitude(tensor: np.ndarray) -> np.ndarray:
    """Converts interleaved IQ data to a complex magnitude vector

    Args:
        tensor (:class:`numpy.ndarray`):
            (vector_length, 2) - sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
    """
    return np.linalg.norm(tensor, axis=1)


def wrapped_phase(tensor: np.ndarray) -> np.ndarray:
    """Converts complex IQ to a wrapped-phase vector

    Args:
        tensor (:class:`numpy.ndarray`):
            vector_length - sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            angle(tensor)
    """
    complex_tensor = tensor[:, 0] + 1j * tensor[:, 1]
    return np.angle(complex_tensor)


def discrete_fourier_transform(tensor: np.ndarray) -> np.ndarray:
    """Computes DFT of complex IQ vector

    Args:
        tensor (:class:`numpy.ndarray`):
             vector_length - sized tensor.

    Returns:
        transformed (:class:`numpy.ndarray`):
            fft(tensor). normalization is 1/sqrt(n)
    """
    complex_tensor = tensor[:, 0] + 1j * tensor[:, 1]
    return np.fft.fft(complex_tensor, norm="ortho")
