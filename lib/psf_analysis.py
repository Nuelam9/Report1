import numpy as np


def get_OTF_from_PSF(PSF: np.ndarray, bb: int = 15) -> np.ndarray:
    """Compute the Optical Transfer Function (OTF) from a know 
       Pupil Function (or PSF).

    Args:
        PSF (np.ndarray): theoretical Pupil Function computed bitwise,
        bb (int, optional): half side of the squared box used to resize
                            the PSF. Defaults to 15.

    Returns: 
        np.ndarray: OTF shifted in the origin.
    """
    xc, yc = np.where(PSF==PSF.max())
    xc, yc = xc[0], yc[0]
    PSF_cut = PSF[yc-bb:yc+bb, xc-bb:xc+bb]
    OTF = np.fft.fft2(PSF_cut)
    OTF_shift = np.fft.fftshift(OTF)
    return OTF_shift
