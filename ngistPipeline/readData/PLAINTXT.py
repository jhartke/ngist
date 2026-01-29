import logging
import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from printStatus import printStatus

from ngistPipeline.readData import der_snr as der_snr


# ======================================
# Routine to load spectra from plain txt
# ======================================
def readCube(config):
    loggingBlanks = (len(os.path.splitext(os.path.basename(__file__))[0]) + 33) * " "

    # Read spectrum
    printStatus.running("Reading the spectrum")
    logging.info("Reading the spectrum: " + config["GENERAL"]["INPUT"])

    # Reading the cube
    data = np.genfromtxt(config["GENERAL"]["INPUT"])
    spec = np.zeros((data.shape[0], 1))
    espec = np.zeros((data.shape[0], 1))
    wave = data[:, 0]
    spec[:, 0] = data[:, 1]
    if data.shape[1] == 3:
        espec[:, 0] = data[:, 2]
    else:
        logging.info(
            "No error extension found. Estimating the error spectra with the der_snr algorithm"
        )
        espec = np.zeros(spec.shape)
        espec[:, 0] = der_snr.der_snr(spec[:, 0])**2*np.ones_like(spec[:, 0])

    # Getting the spatial coordinates
    x = np.zeros(1)
    y = np.zeros(1)
    pixelsize = 1.0

    # De-redshift spectra
    wave = wave / (1 + config["GENERAL"]["REDSHIFT"])
    logging.info(
        "Shifting spectra to rest-frame, assuming a redshift of "
        + str(config["GENERAL"]["REDSHIFT"])
    )

    # Shorten spectra to required wavelength range
    lmin = config["READ_DATA"]["LMIN_TOT"]
    lmax = config["READ_DATA"]["LMAX_TOT"]
    idx = np.where(np.logical_and(wave >= lmin, wave <= lmax))[0]
    spec = spec[idx]
    espec = espec[idx]
    wave = wave[idx]
    logging.info(
        "Shortening spectra to the wavelength range from "
        + str(config["READ_DATA"]["LMIN_TOT"])
        + "A to "
        + str(config["READ_DATA"]["LMAX_TOT"])
        + "A."
    )

    # Computing the SNR per spaxel
    idx_snr = np.where(
        np.logical_and(
            wave >= config["READ_DATA"]["LMIN_SNR"],
            wave <= config["READ_DATA"]["LMAX_SNR"],
        )
    )[0]
    signal = np.zeros(1) + np.nanmedian(spec[idx_snr], axis=0)
    noise = np.zeros(1) + np.abs(np.nanmedian(np.sqrt(espec[idx_snr]), axis=0))
    snr = signal / noise
    logging.info(
        "Computing the signal-to-noise ratio in the wavelength range from "
        + str(config["READ_DATA"]["LMIN_SNR"])
        + "A to "
        + str(config["READ_DATA"]["LMAX_SNR"])
        + "A."
    )
    def create_muse_like_header(x,y):
        """
        Create a mock WCS header for a MUSE-like data cube with 1x1 spatial pixels and a full spectrum.
        
        Parameters:
            n_wavelengths (int): Number of wavelengths in the spectrum.
            lambda_start (float): Starting wavelength in Ångströms (e.g., 4750 for MUSE).
            lambda_step (float): Wavelength step in Ångströms (e.g., spectral resolution like ~1.25 Å/pixel).
            
        Returns:
            Header: Generated FITS header with WCS information for a cube.
        """

        # Initialize a WCS object for 3D data: RA, Dec, Lambda
        wcs = WCS(naxis=2)

        # Spatial axes (RA and Dec) with 1x1 pixels
        wcs.wcs.crpix[0] = x  # Reference pixel for RA (center of the single pixel)
        wcs.wcs.crpix[1] = y  # Reference pixel for Dec
        
        wcs.wcs.cdelt[0] = -0.2 / 3600  # Pixel scale in degrees/pixel for RA (- for increasing RA)
        wcs.wcs.cdelt[1] = 0.2 / 3600   # Pixel scale in degrees/pixel for Dec
        wcs.wcs.crval[0] = 150.0        # Reference value for RA (degrees)
        wcs.wcs.crval[1] = 2.5          # Reference value for Dec (degrees)
        wcs.wcs.ctype[0] = "RA---TAN"   # Projection type for RA
        wcs.wcs.ctype[1] = "DEC--TAN"   # Projection type for Dec

        # Convert WCS object to a FITS header
        header = wcs.to_header()

        # Add additional MUSE-like metadata to the header (optional)
        header['NAXIS1'] = 1              # Number of pixels along RA
        header['NAXIS2'] = 1              # Number of pixels along Dec

        return header

    mock_header = create_muse_like_header(0,0)

    wcshdr = WCS(mock_header).to_header()
    hdr0 = mock_header

    wcshdr = WCS(mock_header).to_header()
    hdr0 = mock_header
    # Storing everything into a structure
    cube = {
        "x": x,
        "y": y,
        "wave": wave,
        "spec": spec,
        "error": espec,
        "snr": snr,
        "signal": signal,
        "noise": noise,
        "pixelsize": pixelsize,
        "wcshdr": wcshdr,
        "hdr0": hdr0,
    }

    printStatus.updateDone("Reading the spectrum")
    logging.info("Finished reading the spectrum!")

    return cube
