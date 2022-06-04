import os
import astropy
import numpy as np
import matplotlib as mpl
from astropy.io import fits
import matplotlib.pyplot as plt
from typing import Tuple


fits_type = astropy.io.fits.hdu.hdulist.HDUList
def get_fits_file(wavelen: str, fits_path: str) -> Tuple[fits_type,
                                                         np.ndarray]:
    """Get the fits obj and the corresponding image. 

    Args:
        wavelen (str): wavelength relative to the fits,
                       can be ['F606w', 'F814w'];
        fits_path (str): path to the fits's directory;

    Returns:
        Tuple[fits_type, np.ndarray]: fits obj and image.
    """
    files = os.listdir(fits_path)
    file = [file for file in files if file.split('_')[-3] == wavelen][0]
    fits_obj = fits.open(fits_path + file)
    image = fits_obj[0].data
    return fits_obj, image


def latex_settings(nrows=1, ncols=1, height=1., length=1.):
    fig, ax = plt.subplots(nrows, ncols, constrained_layout=True)  
    fig_width_pt = 390.0    # Get this from LaTeX using \the\columnwidth
    inches_per_pt = 1.0 / 72.27                          # Convert pt to inches
    golden_mean = (np.sqrt(5) - 1.0) / 2.0               # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt * length    #  width in inches
    fig_height = fig_width * golden_mean * height        # height in inches
    fig_size = [fig_width, fig_height]
    params = {'backend': 'ps',
              'axes.labelsize': 14,
              'legend.fontsize': 9,
              'xtick.labelsize': 10, 
              'ytick.labelsize': 10, 
              'figure.figsize': fig_size,  
              'axes.axisbelow': True}

    mpl.rcParams.update(params)
    return fig, ax


def fancy_legend(leg):
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
        lh.set_linewidth(1)
   

def latex_table_generator(df, filepath, float_format=None, index=False):
    n_cols = len(df.columns)
    latex_cols = [r'\textbf{%s}' %col for col in df.columns]
    with open(filepath, 'w') as tf:
        tf.write(
            df.to_latex(
                index=index,
                float_format=float_format,
                column_format='c' * (n_cols + 1 * index),
                header=latex_cols,
                escape=False
                )
        )
        

def r_squared(y_true, y_pred):
    correlation_matrix = np.corrcoef(y_true, y_pred)
    correlation_xy = correlation_matrix[0,1]
    return correlation_xy ** 2


def Gaussian1D(x, A, mu, var):
    factor = A / np.sqrt(2. * np.pi * var)
    return factor * np.exp(- 0.5 * (x - mu) ** 2. / var)
