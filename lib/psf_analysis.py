import numpy as np


def cut_image(image: np.ndarray, bb: int) -> np.ndarray:
    """Get the image cutted in a square matrix of dimension 2*bbx2*bb.

    Args:
        image (np.ndarray): image to resize,
        bb (int): half side of the squared box used to resize the image.

    Returns:
        np.ndarray: image cutted
    """
    xc, yc = np.where(image==image.max())
    xc, yc = xc[0], yc[0]
    image_cut = image[yc-bb:yc+bb, xc-bb:xc+bb]
    return image_cut    


def get_OTF_from_PSF(PSF: np.ndarray, bb: int = 15) -> np.ndarray:
    """Compute the Optical Transfer Function (OTF) from a know 
       Pupil Function cutted.

    Args:
        PSF (np.ndarray): theoretical Pupil Function computed bitwise,
        bb (int, optional): half side of the squared box used to resize
                            the PSF. Defaults to 15.

    Returns: 
        np.ndarray: OTF shifted in the origin.
    """
    PSF_cut = cut_image(PSF, bb)
    OTF = np.fft.fft2(PSF_cut)
    OTF_shift = np.fft.fftshift(OTF)
    return OTF_shift
