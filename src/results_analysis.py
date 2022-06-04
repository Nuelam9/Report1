import pandas as pd
import numpy as np
import os
import sys
sys.path.append('../lib/')
from utils import get_fits_file, r_squared, Gaussian1D
from multiprocessing import Pool, cpu_count
from astropy.modeling import models, fitting
from typing import Tuple


if __name__ == '__main__':
    filepath_data = '../Data/FITS/'
    wavelen='F606w'
    # Get images as 2-D numpy array from FITS file of F606w
    fits, image = get_fits_file(wavelen=wavelen, 
                                fits_path=filepath_data)

    # setting to 0 all negative values
    image[image < 0] = 0
    
    filepath_results = '../Results/'
    #file = f'ID_coords_image_{wavelen}_filt.csv'
    file = f'ID_coords_image_{wavelen}.csv'
    df = pd.read_csv(filepath_results + file)
    file = f'fit_results_{wavelen}.csv'
     
     
    def get_star_box(ID: int, IDs: np.ndarray = df.to_numpy(),
                     img: np.ndarray = image,
                     bb: int = 15, norm=True) -> Tuple[np.array]:
        """Get the box with the star's maximum intensity at center.

        Args:
            ID (float): star ID;
            IDs (np.ndarray, optional): ID of all isolated stars on the
                                        image. Defaults to df.to_numpy().
            img (np.ndarray, optional): data from fits file.
                                        Defaults to image.
            bb (int, optional): half side of the squared box.
                                Defaults to 15.

        Returns:
            np.array: star's box.
        """
        xc, yc = IDs[IDs[:,0] == ID, 1:].flatten()
        mat = img.copy() 
        box = mat[yc - bb:yc + bb, xc - bb: xc + bb]
        Ai = np.max(box)
        xi = xc - bb + np.where(box == Ai)[1][0]
        yi = yc - bb + np.where(box == Ai)[0][0]
        
        # check if max intensity is at central pixel
        if [xc, yc] != [xi, yi]:
            box = mat[yi - bb: yi + bb, xi - bb: xi + bb]
            xc, yc = xi, yi
        
        if norm:    
            box /= Ai
        else:
            pass
        
        return box, xc, yc, Ai


    def gauss2d_fitting(ID: int) -> np.array:
        """Compute the Gaussian 2D function of a given star.

        Args:
            ID (float): star ID;

        Returns:
            np.array: star's ID, initial guess for the fitting parameters,
                      best fit parameters and r^2.
        """    
        box, *params = get_star_box(ID)
        bb = box.shape[0]
        xc, yc, Ai = params
        yp, xp = box.shape 
        y, x = np.mgrid[:yp, :xp]
        
        try:
            # Fitting
            fit = fitting.LevMarLSQFitter()   
            fi = models.Gaussian2D(amplitude=1., x_mean=bb, y_mean=bb,
                                   x_stddev=1., y_stddev=1., theta=0.)
            f = fit(fi, x, y, box)
            # Best fit parameters
            A = f.amplitude[0]
            x0 = xc - bb + f.x_mean[0]
            y0 = yc - bb + f.y_mean[0]
            sigma_x = f.x_stddev[0]
            sigma_y = f.y_stddev[0]
            theta = f.theta[0]
            # Compute the r squared
            r2 = r_squared(box, f(x, y))
            
            return np.array([ID, Ai, xc, yc, A, x0, y0,
                             sigma_x, sigma_y, theta, r2])
        except:
            return np.array([ID] + [0] * 10)


    def gauss1d_fitting(ID):
        from scipy.optimize import curve_fit
        box, *_ = get_star_box(ID=ID, norm=False)
        xmax, ymax = np.where(box == box.max())
        xmax, ymax = xmax[0], ymax[0]
        
        x_profile = box[ymax, :]
        y_profile = box[:, xmax]
        
        n = box.shape[0]
        x = np.arange(n)
        try:
            popt_x, pcov_x = curve_fit(Gaussian1D, x, x_profile)
            popt_y, pcov_y = curve_fit(Gaussian1D, x, y_profile)
            
            A_x, mu_x, var_x = popt_x
            A_y, mu_y, var_y = popt_y
            
            int_x = Gaussian1D(mu_x, *popt_x)
            int_y = Gaussian1D(mu_y, *popt_y)
            
            x_expected = Gaussian1D(x, *popt_x)
            y_expected = Gaussian1D(x, *popt_y)
            
            r2_x = r_squared(x_profile, x_expected)
            r2_y = r_squared(y_profile, y_expected)
            
            return np.array([ID, int_x, int_y, var_x, var_y, r2_x, r2_y])
        except:
            return np.array([ID] + [np.nan] * 6)


    from psf_analysis import get_OTF_from_PSF
    filepath = '../Results/Hubble/'
    PSF = np.load(filepath + 'Hubble_pupil_PSF.npy')
    OTF_shift = get_OTF_from_PSF(PSF)


    def deconvolution_star(ID: int,
                           OTF_shift: np.ndarray=OTF_shift) -> np.ndarray:
        """Compute the Gaussian 2D function of a given star.

        Args:
            ID (float): star ID;
            IDs (np.ndarray, optional): ID of all isolated stars on the
                                        image. Defaults to df.to_numpy().
            img (np.ndarray, optional): data from fits file.
                                        Defaults to image.

        Returns:
            np.array: star's ID, initial guess for the fitting parameters,
                    best fit parameters and mse.
        """
        box, *_ = get_star_box(ID)
        box_fft = np.fft.fft2(box)
        box_fft_shift = np.fft.fftshift(box_fft)
        
        real_obj_F = box_fft_shift / OTF_shift
        real_obj_shift = np.fft.ifft2(real_obj_F)
        real_obj = np.abs(np.fft.ifftshift(real_obj_shift))
        r2 = r_squared(real_obj, box)
        return np.array([ID, r2])


    def get_save_results(wavelen, fit_func):
        func_name = fit_func.__name__
        name = func_name.split('_')[0]
        file = f'fit_results_{wavelen}_{name}.csv'
        if os.path.isfile(filepath_results + file):
            pass
        else:
            if func_name == 'gauss2d_fitting':
                columns = ['ID', 'Ai', 'xi', 'yi', 'A', 'x0','y0',
                        'sigma_x', 'sigma_y', 'theta', 'r2']
            elif func_name == 'gauss1d_fitting':
                columns = ['ID', 'max_int_x', 'max_int_y',
                        'var_x', 'var_y', 'r2_x', 'r2_y']
            elif func_name == 'deconvolution_star':
                columns = ['ID', 'r2']
            else:
                pass   
            p = Pool(cpu_count())
            res = np.asarray(p.map(fit_func, df.ID.to_numpy()))
            p.close()
            results = pd.DataFrame(res, columns=columns)
            results.to_csv(filepath_results + file, index=False)

    import time
    t1 = time.time()

    get_save_results(wavelen, gauss1d_fitting)
   
    print(time.time() - t1)
