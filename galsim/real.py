# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#
"""@file real.py
Functions for dealing with RealGalaxy objects and the catalogs that store their data.

The RealGalaxy uses images of galaxies from real astrophysical data (e.g. the Hubble Space
Telescope), along with a PSF model of the optical properties of the telescope that took these
images, to simulate new galaxy images with a different (must be larger) telescope PSF.  A
description of the simulation method can be found in Section 5 of Mandelbaum et al. (2012; MNRAS,
540, 1518), although note that the details of the implementation in Section 7 of that work are not
relevant to the more recent software used here.

This module defines the RealGalaxyCatalog class, used to store all required information about a
real galaxy simulation training sample and accompanying PSF model.  For information about
downloading GalSim-readable RealGalaxyCatalog data in FITS format, see the RealGalaxy Data Download
page on the GalSim Wiki: https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy%20Data
"""


import galsim
from galsim import GSObject
from .chromatic import ChromaticSum
import os
import numpy as np

HST_area = 45238.93416  # Area of HST primary mirror in cm^2 from Synphot User's Guide.

# Currently, have bandpasses available for HST COSMOS, AEGIS, and CANDELS.
# ACS zeropoints (AB magnitudes) from
# http://www.stsci.edu/hst/acs/analysis/zeropoints/old_page/localZeropoints#tablestart
# WFC3 zeropoints (AB magnitudes) from
# http://www.stsci.edu/hst/wfc3/phot_zp_lbn
# Format of dictionary entry is:
#    'KEY' : tuple(bandpass filename, zeropoint)
real_galaxy_bandpasses = {
        'F275W': ('WFC3_uvis_F275W.dat', 24.1305),
        'F336W': ('WFC3_uvis_F336W.dat', 24.6682),
        'F435W': ('ACS_wfc_F435W.dat', 25.65777),
        'F606W': ('ACS_wfc_F606W.dat', 26.49113),
        'F775W': ('ACS_wfc_F775W.dat', 25.66504),
        'F814W': ('ACS_wfc_F814W.dat', 25.94333),
        'F850LP': ('ACS_wfc_F850LP.dat', 24.84245),
        'F105W': ('WFC3_ir_F105W.dat', 26.2687),
        'F125W': ('WFC3_ir_F125W.dat', 26.2303),
        'F160W': ('WFC3_ir_F160W.dat', 25.9463)
}

class RealGalaxy(GSObject):
    """A class describing real galaxies from some training dataset.  Its underlying implementation
    uses a Convolution instance of an InterpolatedImage (for the observed galaxy) with a
    Deconvolution of another InterpolatedImage (for the PSF).

    This class uses a catalog describing galaxies in some training data (for more details, see the
    RealGalaxyCatalog documentation) to read in data about realistic galaxies that can be used for
    simulations based on those galaxies.  Also included in the class is additional information that
    might be needed to make or interpret the simulations, e.g., the noise properties of the training
    data.  Users who wish to draw RealGalaxies that have well-defined flux scalings in various
    passbands, and/or parametric representations, should use the COSMOSGalaxy class.

    Because RealGalaxy involves a Deconvolution, `method = 'phot'` is unavailable for the
    drawImage() function.

    Initialization
    --------------

        >>> real_galaxy = galsim.RealGalaxy(real_galaxy_catalog, index=None, id=None, random=False,
        ...                                 rng=None, x_interpolant=None, k_interpolant=None,
        ...                                 flux=None, pad_factor=4, noise_pad_size=0,
        ...                                 gsparams=None)

    This initializes `real_galaxy` with three InterpolatedImage objects (one for the deconvolved
    galaxy, and saved versions of the original HST image and PSF). Note that there are multiple
    keywords for choosing a galaxy; exactly one must be set.

    Note that tests suggest that for optimal balance between accuracy and speed, `k_interpolant` and
    `pad_factor` should be kept at their default values.  The user should be aware that significant
    inaccuracy can result from using other combinations of these parameters; more details can be
    found in http://arxiv.org/abs/1401.2636, especially table 1, and in comment
    https://github.com/GalSim-developers/GalSim/issues/389#issuecomment-26166621 and the following
    comments.

    If you don't set a flux, the flux of the returned object will be the flux of the original
    HST data, scaled to correspond to a 1 second HST exposure (though see the `normalize_area`
    parameter below, and also caveats related to using the `flux` parameter).  If you want a flux
    appropriate for a longer exposure, or for a telescope with a different collecting area than HST,
    you can either renormalize the object with the `flux_rescale` parameter, or by using the
    `exptime` and `area` parameters to `drawImage`.

    Note that RealGalaxy objects use arcsec for the units of their linear dimension.  If you
    are using a different unit for other things (the PSF, WCS, etc.), then you should dilate
    the resulting object with `gal.dilate(galsim.arcsec / scale_unit)`.

    @param real_galaxy_catalog  A RealGalaxyCatalog object with basic information about where to
                            find the data, etc.
    @param index            Index of the desired galaxy in the catalog. [One of `index`, `id`, or
                            `random` is required.]
    @param id               Object ID for the desired galaxy in the catalog. [One of `index`, `id`,
                            or `random` is required.]
    @param random           If True, then select a random galaxy from the catalog.  If the catalog
                            has a 'weight' associated with it to allow for correction of selection
                            effects in which galaxies were included, the 'weight' factor is used to
                            remove those selection effects rather than selecting a completely random
                            object.
                            [One of `index`, `id`, or `random` is required.]
    @param rng              A random number generator to use for selecting a random galaxy
                            (may be any kind of BaseDeviate or None) and to use in generating
                            any noise field when padding.  This user-input random number
                            generator takes precedence over any stored within a user-input
                            CorrelatedNoise instance (see `noise_pad` parameter below).
                            [default: None]
    @param x_interpolant    Either an Interpolant instance or a string indicating which real-space
                            interpolant should be used.  Options are 'nearest', 'sinc', 'linear',
                            'cubic', 'quintic', or 'lanczosN' where N should be the integer order
                            to use. [default: galsim.Quintic()]
    @param k_interpolant    Either an Interpolant instance or a string indicating which k-space
                            interpolant should be used.  Options are 'nearest', 'sinc', 'linear',
                            'cubic', 'quintic', or 'lanczosN' where N should be the integer order
                            to use.  We strongly recommend leaving this parameter at its default
                            value; see text above for details.  [default: galsim.Quintic()]
    @param flux             Total flux, if None then original flux in image is adopted without
                            change.  Note that, technically, this parameter sets the flux of the
                            postage stamp image and not the flux of the contained galaxy.  These two
                            values will be strongly correlated when the signal-to-noise ratio of the
                            galaxy is large, but may be considerably different if the flux of the
                            galaxy is small with respect to the noise variations in the postage
                            stamp.  To avoid complications with faint galaxies, consider using the
                            flux_rescale parameter.  [default: None]
    @param flux_rescale     Flux rescaling factor; if None, then no rescaling is done.  Either
                            `flux` or `flux_rescale` may be set, but not both. [default: None]
    @param pad_factor       Factor by which to pad the Image when creating the
                            InterpolatedImage.  We strongly recommend leaving this parameter
                            at its default value; see text above for details.  [default: 4]
    @param noise_pad_size   If provided, the image will be padded out to this size (in arcsec)
                            with the noise specified in the real galaxy catalog. This is
                            important if you are planning to whiten the resulting image.  You
                            should make sure that the padded image is larger than the postage
                            stamp onto which you are drawing this object.
                            [default: None]
    @param normalize_area   By default, the flux of the returned object is normalized such that
                            drawing with exptime=1 and area=1 (the `drawImage` defaults) simulates
                            an image with the appropriate number of counts for a 1 second HST
                            exposure.  Setting this keyword to True will instead normalize the
                            returned object such that to simulate a 1 second HST exposure, you must
                            use drawImage with exptime=1 and area=45238.93416 (the HST collecting
                            area in cm^2).  [default: False]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]
    @param logger           A logger object for output of progress statements if the user wants
                            them.  [default: None]

    Methods
    -------

    There are no additional methods for RealGalaxy beyond the usual GSObject methods.
    """
    _req_params = {}
    _opt_params = { "x_interpolant" : str ,
                    "k_interpolant" : str ,
                    "flux" : float ,
                    "flux_rescale" : float ,
                    "pad_factor" : float,
                    "noise_pad_size" : float,
                    "normalize_area" : bool
                  }
    _single_params = [ { "index" : int , "id" : str , "random" : bool } ]
    _takes_rng = True

    def __init__(self, real_galaxy_catalog, index=None, id=None, random=False,
                 rng=None, x_interpolant=None, k_interpolant=None, flux=None, flux_rescale=None,
                 pad_factor=4, noise_pad_size=0, normalize_area=False, gsparams=None, logger=None):


        if rng is None:
            self.rng = galsim.BaseDeviate()
        elif not isinstance(rng, galsim.BaseDeviate):
            raise TypeError("The rng provided to RealGalaxy constructor is not a BaseDeviate")
        else:
            self.rng = rng
        self._rng = self.rng.duplicate()  # This is only needed if we want to make sure eval(repr)
                                          # results in the same object.

        if flux is not None and flux_rescale is not None:
            raise TypeError("Cannot supply a flux and a flux rescaling factor!")

        if isinstance(real_galaxy_catalog, tuple):
            # Special (undocumented) way to build a RealGalaxy without needing the rgc directly
            # by providing the things we need from it.  Used by COSMOSGalaxy.
            self.gal_image, self.psf_image, noise_image, pixel_scale, var = real_galaxy_catalog
            use_index = 0  # For the logger statements below.
            if logger:
                logger.debug('RealGalaxy %d: Start RealGalaxy constructor.',use_index)
            self.catalog_file = None
        else:
            # Get the index to use in the catalog
            if index is not None:
                if id is not None or random:
                    raise AttributeError('Too many methods for selecting a galaxy!')
                use_index = index
            elif id is not None:
                if random:
                    raise AttributeError('Too many methods for selecting a galaxy!')
                use_index = real_galaxy_catalog.getIndexForID(id)
            elif random:
                ud = galsim.UniformDeviate(self.rng)
                use_index = int(real_galaxy_catalog.nobjects * ud())
                if hasattr(real_galaxy_catalog, 'weight'):
                    # If weight factors are available, make sure the random selection uses the
                    # weights to remove the catalog-level selection effects (flux_radius-dependent
                    # probability of making a postage stamp for a given object).
                    while ud() > real_galaxy_catalog.weight[use_index]:
                        # Pick another one to try.
                        use_index = int(real_galaxy_catalog.nobjects * ud())
            else:
                raise AttributeError('No method specified for selecting a galaxy!')
            if logger:
                logger.debug('RealGalaxy %d: Start RealGalaxy constructor.',use_index)

            # Read in the galaxy, PSF images; for now, rely on pyfits to make I/O errors.
            self.gal_image = real_galaxy_catalog.getGalImage(use_index)
            if logger:
                logger.debug('RealGalaxy %d: Got gal_image',use_index)

            self.psf_image = real_galaxy_catalog.getPSFImage(use_index)
            if logger:
                logger.debug('RealGalaxy %d: Got psf_image',use_index)

            #self.noise = real_galaxy_catalog.getNoise(use_index, self.rng, gsparams)
            # We need to duplication some of the RealGalaxyCatalog.getNoise() function, since we
            # want it to be possible to have the RealGalaxyCatalog in another process, and the
            # BaseCorrelatedNoise object is not picklable.  So we just build it here instead.
            noise_image, pixel_scale, var = real_galaxy_catalog.getNoiseProperties(use_index)
            if logger:
                logger.debug('RealGalaxy %d: Got noise_image',use_index)
            self.catalog_file = real_galaxy_catalog.getFileName()

        if noise_image is None:
            self.noise = galsim.UncorrelatedNoise(var, rng=self.rng, scale=pixel_scale,
                                                  gsparams=gsparams)
        else:
            ii = galsim.InterpolatedImage(noise_image, normalization="sb",
                                          calculate_stepk=False, calculate_maxk=False,
                                          x_interpolant='linear', gsparams=gsparams)
            self.noise = galsim.correlatednoise._BaseCorrelatedNoise(self.rng, ii, noise_image.wcs)
            self.noise = self.noise.withVariance(var)
        if logger:
            logger.debug('RealGalaxy %d: Finished building noise',use_index)

        # Save any other relevant information as instance attributes
        self.catalog = real_galaxy_catalog
        self.index = use_index
        self.pixel_scale = float(pixel_scale)
        self._x_interpolant = x_interpolant
        self._k_interpolant = k_interpolant
        self._pad_factor = pad_factor
        self._noise_pad_size = noise_pad_size
        self._flux = flux
        self._flux_rescale = flux_rescale
        self._normalize_area = normalize_area
        self._gsparams = gsparams

        # Convert noise_pad to the right noise to pass to InterpolatedImage
        if noise_pad_size:
            noise_pad = self.noise
        else:
            noise_pad = 0.

        # Build the InterpolatedImage of the PSF.
        self.original_psf = galsim.InterpolatedImage(
            self.psf_image, x_interpolant=x_interpolant, k_interpolant=k_interpolant,
            flux=1.0, gsparams=gsparams)
        if logger:
            logger.debug('RealGalaxy %d: Made original_psf',use_index)

        # Build the InterpolatedImage of the galaxy.
        # Use the stepK() value of the PSF as a maximum value for stepK of the galaxy.
        # (Otherwise, low surface brightness galaxies can get a spuriously high stepk, which
        # leads to problems.)
        self.original_gal = galsim.InterpolatedImage(
                self.gal_image, x_interpolant=x_interpolant, k_interpolant=k_interpolant,
                pad_factor=pad_factor, noise_pad_size=noise_pad_size,
                calculate_stepk=self.original_psf.stepK(),
                calculate_maxk=self.original_psf.maxK(),
                noise_pad=noise_pad, rng=self.rng, gsparams=gsparams)
        if logger:
            logger.debug('RealGalaxy %d: Made original_gal',use_index)

        # Only alter normalization if a change is requested
        if flux is not None or flux_rescale is not None or normalize_area:
            if flux_rescale is None:
                flux_rescale = 1
            if normalize_area:
                flux_rescale /= HST_area
            if flux is not None:
                flux_rescale *= flux/self.original_gal.getFlux()
            self.original_gal *= flux_rescale
            self.noise *= flux_rescale**2

        # Calculate the PSF "deconvolution" kernel
        psf_inv = galsim.Deconvolve(self.original_psf, gsparams=gsparams)

        # Initialize the SBProfile attribute
        GSObject.__init__(
            self, galsim.Convolve([self.original_gal, psf_inv], gsparams=gsparams).SBProfile)
        if logger:
            logger.debug('RealGalaxy %d: Made gsobject',use_index)

        # Save the noise in the image as an accessible attribute
        self.noise = self.noise.convolvedWith(psf_inv, gsparams)
        if logger:
            logger.debug('RealGalaxy %d: Finished building RealGalaxy',use_index)

    def getHalfLightRadius(self):
        raise NotImplementedError("Half light radius calculation not implemented for RealGalaxy "
                                   +"objects.")

    def __eq__(self, other):
        return (isinstance(other, galsim.RealGalaxy) and
                self.catalog == other.catalog and
                self.index == other.index and
                self._x_interpolant == other._x_interpolant and
                self._k_interpolant == other._k_interpolant and
                self._pad_factor == other._pad_factor and
                self._noise_pad_size == other._noise_pad_size and
                self._flux == other._flux and
                self._flux_rescale == other._flux_rescale and
                self._normalize_area == other._normalize_area and
                self._rng == other._rng and
                self._gsparams == other._gsparams)

    def __hash__(self):
        return hash(("galsim.RealGalaxy", self.catalog, self.index, self._x_interpolant,
                     self._k_interpolant, self._pad_factor, self._noise_pad_size, self._flux,
                     self._flux_rescale, self._normalize_area, self._rng.serialize(), self._gsparams))

    def __repr__(self):
        s = 'galsim.RealGalaxy(%r, index=%r, '%(self.catalog, self.index)
        if self._x_interpolant is not None:
            s += 'x_interpolant=%r, '%self._x_interpolant
        if self._k_interpolant is not None:
            s += 'k_interpolant=%r, '%self._k_interpolant
        if self._pad_factor != 4:
            s += 'pad_factor=%r, '%self._pad_factor
        if self._noise_pad_size != 0:
            s += 'noise_pad_size=%r, '%self._noise_pad_size
        if self._flux is not None:
            s += 'flux=%r, '%self._flux
        if self._flux_rescale is not None:
            s += 'flux_rescale=%r, '%self._flux_rescale
        if self._normalize_area:
            s += 'normalize_area=True, '
        s += 'rng=%r, '%self._rng
        s += 'gsparams=%r)'%self._gsparams
        return s

    def __str__(self):
        # I think this is more intuitive without the RealGalaxyCatalog parameter listed.
        return 'galsim.RealGalaxy(index=%s, flux=%s)'%(self.index, self.flux)

    def __getstate__(self):
        # The SBProfile is picklable, but it is pretty inefficient, due to the large images being
        # written as a string.  Better to pickle the image and remake the InterpolatedImage.
        d = self.__dict__.copy()
        del d['SBProfile']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        psf_inv = galsim.Deconvolve(self.original_psf, gsparams=self._gsparams)
        GSObject.__init__(
            self, galsim.Convolve([self.original_gal, psf_inv], gsparams=self._gsparams).SBProfile)



class RealGalaxyCatalog(object):
    """Class containing a catalog with information about real galaxy training data.

    The RealGalaxyCatalog class reads in and stores information about a specific training sample of
    realistic galaxies. We assume that all files containing the images (galaxies and PSFs) live in
    one directory; they could be individual files, or multiple HDUs of the same file.  Currently
    there is no functionality that lets this be a FITS data cube, because we assume that the object
    postage stamps will in general need to be different sizes depending on the galaxy size.

    Note that when simulating galaxies based on HST but using either realistic or parametric galaxy
    models, the COSMOSCatalog class may be more useful.  It allows the imposition of selection
    criteria and other subtleties that are more difficult to impose with RealGalaxyCatalog.

    While you could create your own catalog to use with this class, the typical use cases would
    be to use one of the catalogs that we have created and distributed.  There are three such
    catalogs currently, which can be use with one of the following initializations:

    1. A small example catalog is distributed with the GalSim distribution.  This catalog only
       has 100 galaxies, so it is not terribly useful as a representative galaxy population.
       But for simplistic use cases, it might be sufficient.  We use it for our unit tests and
       in some of the demo scripts (demo6, demo10, and demo11).  To use this catalog, you would
       initialize with

           >>> rgc = galsim.RealGalaxyCatalog('real_galaxy_catalog_23.5_example.fits',
                                              dir='path/to/GalSim/examples/data')

    2. There are two larger catalogs based on HST observations of the COSMOS field with around
       26,000 and 56,000 galaxies each with a limiting magnitude of F814W=23.5.  (The former is
       a subset of the latter.) For information about how to download these catalogs, see the
       RealGalaxy Data Download Page on the GalSim Wiki:

           https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy%20Data

       Be warned that the catalogs are quite large.  The larger one is around 11 GB after unpacking
       the tarball.  To use one of these catalogs, you would initialize with

           >>> rgc = galsim.RealGalaxyCatalog('real_galaxy_catalog_23.5.fits',
                                              dir='path/to/download/directory')

    3. There is a catalog containing a random subsample of the HST COSMOS images with a limiting
       magnitude of F814W=25.2.  More information about downloading these catalogs can be found on
       the RealGalaxy Data Download page linked above.

    4. Finally, we provide a program that will download the large COSMOS sample for you and
       put it in the $PREFIX/share/galsim directory of your installation path.  The program is

           galsim_download_cosmos

       which gets installed in the $PREFIX/bin directory when you install GalSim.  If you use
       this program to download the COSMOS catalog, then you can use it with

           >>> rgc = galsim.RealGalaxyCatalog()

       GalSim knows the location of the installation share directory, so it will automatically
       look for it there.

    @param file_name  The file containing the catalog. [default: None, which will look for the
                      F814W<25.2 COSMOS catalog in $PREFIX/share/galsim.  It will raise an
                      exception if the catalog is not there telling you to run
                      galsim_download_cosmos.]
    @param sample     A keyword argument that can be used to specify the sample to use, i.e.,
                      "23.5" or "25.2".  At most one of `file_name` and `sample` should be
                      specified.
                      [default: None, which results in the same default as `file_name=None`.]
    @param dir        The directory containing the catalog, image, and noise files, or symlinks to
                      them. [default: None]
    @param preload    Whether to preload the header information.  If `preload=True`, the bulk of
                      the I/O time is in the constructor.  If `preload=False`, there is
                      approximately the same total I/O time (assuming you eventually use most of
                      the image files referenced in the catalog), but it is spread over the
                      various calls to getGalImage() and getPSFImage().  [default: False]
    @param logger     An optional logger object to log progress. [default: None]
    """
    _req_params = {}
    _opt_params = { 'file_name' : str, 'sample' : str, 'dir' : str,
                    'preload' : bool }
    _single_params = []
    _takes_rng = False

    # _nobject_only is an intentionally undocumented kwarg that should be used only by
    # the config structure.  It indicates that all we care about is the nobjects parameter.
    # So skip any other calculations that might normally be necessary on construction.
    def __init__(self, file_name=None, sample=None, image_dir=None, dir=None, preload=False,
                 noise_dir=None, logger=None, _nobjects_only=False):
        if sample is not None and file_name is not None:
            raise ValueError("Cannot specify both the sample and file_name!")

        from galsim._pyfits import pyfits
        self.file_name, self.image_dir, self.noise_dir, _ = \
            _parse_files_dirs(file_name, image_dir, dir, noise_dir, sample)

        with pyfits.open(self.file_name) as fits:
            self.cat = fits[1].data
        self.nobjects = len(self.cat) # number of objects in the catalog
        if _nobjects_only: return  # Exit early if that's all we needed.
        ident = self.cat.field('ident') # ID for object in the training sample

        # We want to make sure that the ident array contains all strings.
        # Strangely, ident.astype(str) produces a string with each element == '1'.
        # Hence this way of doing the conversion:
        self.ident = [ "%s"%val for val in ident ]

        self.gal_file_name = self.cat.field('gal_filename') # file containing the galaxy image
        self.psf_file_name = self.cat.field('PSF_filename') # file containing the PSF image

        # Add the directories:
        # Note the strip call.  Sometimes the filenames have an extra space at the end.
        # This gets rid of that space.
        self.gal_file_name = [ os.path.join(self.image_dir,f.strip()) for f in self.gal_file_name ]
        self.psf_file_name = [ os.path.join(self.image_dir,f.strip()) for f in self.psf_file_name ]

        # We don't require the noise_filename column.  If it is not present, we will use
        # Uncorrelated noise based on the variance column.
        try:
            self.noise_file_name = self.cat.field('noise_filename') # file containing the noise cf
            self.noise_file_name = [ os.path.join(self.noise_dir,f) for f in self.noise_file_name ]
        except:
            self.noise_file_name = None

        self.gal_hdu = self.cat.field('gal_hdu') # HDU containing the galaxy image
        self.psf_hdu = self.cat.field('PSF_hdu') # HDU containing the PSF image
        self.pixel_scale = self.cat.field('pixel_scale') # pixel scale for image (could be different
        # if we have training data from other datasets... let's be general here and make it a
        # vector in case of mixed training set)
        self.variance = self.cat.field('noise_variance') # noise variance for image
        self.mag = self.cat.field('mag')   # apparent magnitude
        self.band = self.cat.field('band') # bandpass in which apparent mag is measured, e.g., F814W
        # The weight factor should be a float value >=0 (so that random selections of indices can
        # use it to remove any selection effects in the catalog creation process).
        # Here we renormalize by the maximum weight.  If the maximum is below 1, that just means
        # that all galaxies were subsampled at some level, and here we only want to account for
        # relative selection effects within the catalog, not absolute subsampling.  If the maximum
        # is above 1, then our random number generation test used to draw a weighted sample will
        # fail since we use uniform deviates in the range 0 to 1.
        weight = self.cat.field('weight')
        self.weight = weight/np.max(weight)
        if 'stamp_flux' in self.cat.names:
            self.stamp_flux = self.cat.field('stamp_flux')

        self.saved_noise_im = {}
        self.loaded_files = {}
        self.logger = logger

        # The pyfits commands aren't thread safe.  So we need to make sure the methods that
        # use pyfits are not run concurrently from multiple threads.
        from multiprocessing import Lock
        self.gal_lock = Lock()  # Use this when accessing gal files
        self.psf_lock = Lock()  # Use this when accessing psf files
        self.loaded_lock = Lock()  # Use this when opening new files from disk
        self.noise_lock = Lock()  # Use this for building the noise image(s) (usually just one)

        # Preload all files if desired
        if preload: self.preload()
        self._preload = preload

        # eventually I think we'll want information about the training dataset,
        # i.e. (dataset, ID within dataset)
        # also note: will be adding bits of information, like noise properties and galaxy fit params

    def __del__(self):
        # Make sure to clean up pyfits open files if people forget to call close()
        self.close()

    def close(self):
        # Need to close any open files.
        # Make sure to check if loaded_files exists, since the constructor could abort
        # before it gets to the place where loaded_files is built.
        if hasattr(self, 'loaded_files'):
            for f in self.loaded_files.values():
                f.close()
        self.loaded_files = {}

    def getNObjects(self) : return self.nobjects
    def __len__(self): return self.nobjects
    def getFileName(self) : return self.file_name

    def getIndexForID(self, id):
        """Internal function to find which index number corresponds to the value ID in the ident
        field.
        """
        # Just to be completely consistent, convert id to a string in the same way we
        # did above for the ident array:
        id = "%s"%id
        if id in self.ident:
            return self.ident.index(id)
        else:
            raise ValueError('ID %s not found in list of IDs'%id)

    def preload(self):
        """Preload the files into memory.

        There are memory implications to this, so we don't do this by default.  However, it can be
        a big speedup if memory isn't an issue.  Especially if many (or all) of the images are
        stored in the same file as different HDUs.
        """
        from galsim._pyfits import pyfits
        if self.logger:
            self.logger.debug('RealGalaxyCatalog: start preload')
        for file_name in np.concatenate((self.gal_file_name , self.psf_file_name)):
            # numpy sometimes add a space at the end of the string that is not present in
            # the original file.  Stupid.  But this next line removes it.
            file_name = file_name.strip()
            if file_name not in self.loaded_files:
                if self.logger:
                    self.logger.debug('RealGalaxyCatalog: preloading %s',file_name)
                # I use memmap=False, because I was getting problems with running out of
                # file handles in the great3 real_gal run, which uses a lot of rgc files.
                # I think there must be a bug in pyfits that leaves file handles open somewhere
                # when memmap = True.  Anyway, I don't know what the performance implications
                # are (since I couldn't finish the run with the default memmap=True), but I
                # don't think there is much impact either way with memory mapping in our case.
                self.loaded_files[file_name] = pyfits.open(file_name,memmap=False)

    def _getFile(self, file_name):
        from galsim._pyfits import pyfits
        if file_name in self.loaded_files:
            if self.logger:
                self.logger.debug('RealGalaxyCatalog: File %s is already open',file_name)
            f = self.loaded_files[file_name]
        else:
            self.loaded_lock.acquire()
            # Check again in case two processes both hit the else at the same time.
            if file_name in self.loaded_files: # pragma: no cover
                if self.logger:
                    self.logger.debug('RealGalaxyCatalog: File %s is already open',file_name)
                f = self.loaded_files[file_name]
            else:
                if self.logger:
                    self.logger.debug('RealGalaxyCatalog: open file %s',file_name)
                f = pyfits.open(file_name,memmap=False)
                self.loaded_files[file_name] = f
            self.loaded_lock.release()
        return f

    def getBandpass(self):
        """Returns a Bandpass object for the catalog.
        """
        try:
            bp = real_galaxy_bandpasses[self.band[0].upper()]
        except KeyError:
            raise ValueError("Bandpass not found.  To use bandpass '{0}', please add an entry to "
                             "the galsim.real.real_galaxy_bandpasses "
                             "dictionary.".format(self.band[0]))
        return galsim.Bandpass(bp[0], wave_type='nm', zeropoint=bp[1])

    def getGalImage(self, i):
        """Returns the galaxy at index `i` as an Image object.
        """
        if self.logger:
            self.logger.debug('RealGalaxyCatalog %d: Start getGalImage',i)
        if i >= len(self.gal_file_name):
            raise IndexError(
                'index %d given to getGalImage is out of range (0..%d)'
                % (i,len(self.gal_file_name)-1))
        f = self._getFile(self.gal_file_name[i])
        # For some reason the more elegant `with gal_lock:` syntax isn't working for me.
        # It gives an EOFError.  But doing an explicit acquire and release seems to work fine.
        self.gal_lock.acquire()
        array = f[self.gal_hdu[i]].data
        self.gal_lock.release()
        im = galsim.Image(np.ascontiguousarray(array.astype(np.float64)),
                          scale=self.pixel_scale[i])
        return im

    def getPSFImage(self, i):
        """Returns the PSF at index `i` as an Image object.
        """
        if self.logger:
            self.logger.debug('RealGalaxyCatalog %d: Start getPSFImage',i)
        if i >= len(self.psf_file_name):
            raise IndexError(
                'index %d given to getPSFImage is out of range (0..%d)'
                % (i,len(self.psf_file_name)-1))
        f = self._getFile(self.psf_file_name[i])
        self.psf_lock.acquire()
        array = f[self.psf_hdu[i]].data
        self.psf_lock.release()
        return galsim.Image(np.ascontiguousarray(array.astype(np.float64)),
                            scale=self.pixel_scale[i])

    def getPSF(self, i, x_interpolant=None, k_interpolant=None, gsparams=None):
        """Returns the PSF at index `i` as a GSObject.
        """
        psf_image = self.getPSFImage(i)
        return galsim.InterpolatedImage(psf_image,
                                        x_interpolant=x_interpolant, k_interpolant=k_interpolant,
                                        flux=1.0, gsparams=gsparams)

    def getNoiseProperties(self, i):
        """Returns the components needed to make the noise correlation function at index `i`.
           Specifically, the noise image (or None), the pixel_scale, and the noise variance,
           as a tuple (im, scale, var).
        """

        if self.logger:
            self.logger.debug('RealGalaxyCatalog %d: Start getNoise',i)
        if self.noise_file_name is None:
            im = None
        else:
            if i >= len(self.noise_file_name):
                raise IndexError(
                    'index %d given to getNoise is out of range (0..%d)'%(
                        i,len(self.noise_file_name)-1))
            if self.noise_file_name[i] in self.saved_noise_im:
                im = self.saved_noise_im[self.noise_file_name[i]]
                if self.logger:
                    self.logger.debug('RealGalaxyCatalog %d: Got saved noise im',i)
            else:
                self.noise_lock.acquire()
                # Again, a second check in case two processes get here at the same time.
                if self.noise_file_name[i] in self.saved_noise_im:
                    im = self.saved_noise_im[self.noise_file_name[i]]
                    if self.logger:
                        self.logger.debug('RealGalaxyCatalog %d: Got saved noise im',i)
                else:
                    from galsim._pyfits import pyfits
                    with pyfits.open(self.noise_file_name[i]) as fits:
                        array = fits[0].data
                    im = galsim.Image(np.ascontiguousarray(array.astype(np.float64)),
                                      scale=self.pixel_scale[i])
                    self.saved_noise_im[self.noise_file_name[i]] = im
                    if self.logger:
                        self.logger.debug('RealGalaxyCatalog %d: Built noise im',i)
                self.noise_lock.release()

        return im, self.pixel_scale[i], self.variance[i]

    def getNoise(self, i, rng=None, gsparams=None):
        """Returns the noise correlation function at index `i` as a CorrelatedNoise object.
           Note: the return value from this function is not picklable, so this cannot be used
           across processes.
        """
        im, scale, var = self.getNoiseProperties(i)
        if im is None:
            cf = galsim.UncorrelatedNoise(var, rng=rng, scale=scale, gsparams=gsparams)
        else:
            ii = galsim.InterpolatedImage(im, normalization="sb",
                                          calculate_stepk=False, calculate_maxk=False,
                                          x_interpolant='linear', gsparams=gsparams)
            cf = galsim.correlatednoise._BaseCorrelatedNoise(rng, ii, im.wcs)
            cf = cf.withVariance(var)
        return cf

    def __repr__(self):
        return 'galsim.RealGalaxyCatalog(%r)'%self.file_name

    def __eq__(self, other):
        return (isinstance(other, RealGalaxyCatalog) and
                self.file_name == other.file_name and
                self.image_dir == other.image_dir and
                self.noise_dir == other.noise_dir)
    def __ne__(self, other): return not self.__eq__(other)

    def __hash__(self): return hash(repr(self))

    def __getstate__(self):
        d = self.__dict__.copy()
        d['loaded_files'] = {}
        d['saved_noise_im'] = {}
        del d['gal_lock']
        del d['psf_lock']
        del d['loaded_lock']
        del d['noise_lock']
        return d

    def __setstate__(self, d):
        from multiprocessing import Lock
        self.__dict__ = d
        self.gal_lock = Lock()
        self.psf_lock = Lock()
        self.loaded_lock = Lock()
        self.noise_lock = Lock()
        pass

def simReal(real_galaxy, target_PSF, target_pixel_scale, g1=0.0, g2=0.0, rotation_angle=None,
            rand_rotate=True, rng=None, target_flux=1000.0, image=None): # pragma: no cover
    """Deprecated method to simulate images (no added noise) from real galaxy training data.

    This function takes a RealGalaxy from some training set, and manipulates it as needed to
    simulate a (no-noise-added) image from some lower-resolution telescope.  It thus requires a
    target PSF (which could be an image, or one of our base classes) that represents all PSF
    components including the pixel response, and a target pixel scale.

    The default rotation option is to impose a random rotation to make irrelevant any real shears
    in the galaxy training data (optionally, the RNG can be supplied).  This default can be turned
    off by setting `rand_rotate = False` or by requesting a specific rotation angle using the
    `rotation_angle` keyword, in which case `rand_rotate` is ignored.

    Optionally, the user can specify a shear (default 0).  Finally, they can specify a flux
    normalization for the final image, default 1000.

    @param real_galaxy      The RealGalaxy object to use, not modified in generating the
                            simulated image.
    @param target_PSF       The target PSF, either one of our base classes or an Image.
    @param target_pixel_scale  The pixel scale for the final image, in arcsec.
    @param g1               First component of shear to impose (components defined with respect
                            to pixel coordinates), [default: 0]
    @param g2               Second component of shear to impose, [default: 0]
    @param rotation_angle   Angle by which to rotate the galaxy (must be an Angle
                            instance). [default: None]
    @param rand_rotate      Should the galaxy be rotated by some random angle?  [default: True;
                            unless `rotation_angle` is set, then False]
    @param rng              A BaseDeviate instance to use for the random selection or rotation
                            angle. [default: None]
    @param target_flux      The target flux in the output galaxy image, [default: 1000.]
    @param image            As with the GSObject.drawImage() function, if an image is provided,
                            then it will be used and returned.  [default: None, which means an
                            appropriately-sized image will be created.]

    @return a simulated galaxy image.
    """
    from .deprecated import depr
    depr('simReal', 1.5, '',
         'This method has been deprecated due to lack of widespread use.  If you '+
         'have a need for it, please open an issue requesting that it be reinstated.')
    # do some checking of arguments
    if not isinstance(real_galaxy, galsim.RealGalaxy):
        raise RuntimeError("Error: simReal requires a RealGalaxy!")
    if isinstance(target_PSF, galsim.Image):
        target_PSF = galsim.InterpolatedImage(target_PSF, scale=target_pixel_scale)
    if not isinstance(target_PSF, galsim.GSObject):
        raise RuntimeError("Error: target PSF is not an Image or GSObject!")
    if rotation_angle is not None and not isinstance(rotation_angle, galsim.Angle):
        raise RuntimeError("Error: specified rotation angle is not an Angle instance!")
    if (target_pixel_scale < real_galaxy.pixel_scale):
        import warnings
        message = "Warning: requested pixel scale is higher resolution than original!"
        warnings.warn(message)
    import math # needed for pi, sqrt below
    g = math.sqrt(g1**2 + g2**2)
    if g > 1:
        raise RuntimeError("Error: requested shear is >1!")

    # make sure target PSF is normalized
    target_PSF = target_PSF.withFlux(1.0)

    # rotate
    if rotation_angle is not None:
        real_galaxy = real_galaxy.rotate(rotation_angle)
    elif rotation_angle is None and rand_rotate:
        ud = galsim.UniformDeviate(rng)
        rand_angle = galsim.Angle(math.pi*ud(), galsim.radians)
        real_galaxy = real_galaxy.rotate(rand_angle)

    # set fluxes
    real_galaxy = real_galaxy.withFlux(target_flux)

    # shear
    if (g1 != 0.0 or g2 != 0.0):
        real_galaxy = real_galaxy.shear(g1=g1, g2=g2)

    # convolve, resample
    out_gal = galsim.Convolve([real_galaxy, target_PSF])
    image = out_gal.drawImage(image=image, scale=target_pixel_scale, method='no_pixel')

    # return simulated image
    return image

def _parse_files_dirs(file_name, image_dir, dir, noise_dir, sample):
    if image_dir is not None or noise_dir is not None:  # pragma: no cover
        from .deprecated import depr
        if image_dir is not None:
            depr('image_dir', 1.4, 'dir')
        if noise_dir is not None:
            depr('noise_dir', 1.4, 'dir')

    if sample is None:
        if file_name is None:
            use_sample = '25.2'
        elif '25.2' in file_name:
            use_sample = '25.2'
        elif '23.5' in file_name:
            use_sample = '23.5'
        else:
            use_sample = None
    else:
        use_sample = sample
        if use_sample != '25.2' and use_sample != '23.5':
            raise ValueError("Sample name not recognized: %s"%use_sample)
    # after that piece of code, use_sample is either "23.5", "25.2" (if using one of the default
    # catalogs) or it is still None, if a file_name was given.

    if file_name is None:
        if image_dir is not None:
            raise ValueError('Cannot specify image_dir when using default file_name.')
        file_name = 'real_galaxy_catalog_' + use_sample + '.fits'
        if dir is None:
            dir = os.path.join(galsim.meta_data.share_dir,
                               'COSMOS_'+use_sample+'_training_sample')
        full_file_name = os.path.join(dir,file_name)
        full_image_dir = dir
        if not os.path.isfile(full_file_name):
            raise RuntimeError('No RealGalaxy catalog found in %s.  '%dir +
                               'Run the program galsim_download_cosmos -s %s '%use_sample +
                               'to download catalog and accompanying image files.')
    elif dir is None:
        full_file_name = file_name
        if image_dir is None:
            full_image_dir = os.path.dirname(file_name)
        elif os.path.dirname(image_dir) == '':
            full_image_dir = os.path.join(os.path.dirname(full_file_name),image_dir)
        else:
            full_image_dir = image_dir
    else:
        full_file_name = os.path.join(dir,file_name)
        if image_dir is None:
            full_image_dir = dir
        else:
            full_image_dir = os.path.join(dir,image_dir)
    if not os.path.isfile(full_file_name):
        raise IOError(full_file_name+' not found.')
    if not os.path.isdir(full_image_dir):
        raise IOError(full_image_dir+' directory does not exist!')

    if noise_dir is None:
        full_noise_dir = full_image_dir
    else:
        if not os.path.isdir(noise_dir):
            raise IOError(noise_dir+' directory does not exist!')
        full_noise_dir = noise_dir

    return full_file_name, full_image_dir, full_noise_dir, use_sample


class ChromaticRealGalaxy(ChromaticSum):
    """A class describing real galaxies over multiple wavelengths, using some multi-band training
    dataset.  The underlying implementation models multi-band images of individual galaxies
    as chromatic PSF convolutions (and integrations over wavelength) with a sum of profiles
    separable into spatial and spectral components.  The spectral components are specified by the
    user, and the spatial components are determined one Fourier mode at a time by the class.  This
    decomposition can be thought of as a constrained chromatic deconvolution of the multi-band
    images by the associated PSFs, similar in spirit to RealGalaxy.

    Because ChromaticRealGalaxy involves an InterpolatedKImage, `method = 'phot'` is unavailable for
    the drawImage() function.

    Initialization
    --------------

    Fundamentally, the required inputs for this class are (1) a series of high resolution input
    `Image`s of a single galaxy in different bands, (2) the `Bandpass`es corresponding to those
    images, (3) the PSFs of those images as either `GSObject`s or `ChromaticObject`s, and (4) the
    noise properties of the input images as instances of either `CorrelatedNoise` or
    `UncorrelatedNoise`.  If you want to specify these inputs directly, that is possible via
    the `.makeFromImages` factory method of this class:

        >>> crg = galsim.ChromaticRealGalaxy.makeFromImages(imgs, bands, PSFs, xis, ...)

    Alternatively, you may create a ChromaticRealGalaxy via a list of `RealGalaxyCatalog`s that
    correspond to a set of galaxies observed in different bands:

        >>> crg = galsim.ChromaticRealGalaxy(real_galaxy_catalogs, index=0, ...)

    The above will use the 1st object in the catalogs, which should be the same galaxy, just
    observed in different bands.  Note that there are multiple keywords for choosing a galaxy from
    a catalog; exactly one must be set.  In the future we may add more such options, e.g., to
    choose at random but accounting for the non-constant weight factors (probabilities for
    objects to make it into the training sample).

    The flux normalization of the returned object will by default match the original data, scaled to
    correspond to a 1 second HST exposure (though see the `normalize_area` parameter).  If you want
    a flux appropriate for a longer exposure or telescope with different collecting area, you can
    use the `ChromaticObject` method `withScaledFlux` on the returned object, or use the `exptime`
    and `area` keywords to `drawImage`.  Note that while you can also use the `ChromaticObject`
    methods `withFlux`, `withMagnitude`, and `withFluxDensity` to set the absolute normalization,
    these methods technically adjust the flux of the entire postage stamp image (including noise!)
    and not necessarily the flux of the galaxy itself.  (These two fluxes will be strongly
    correlated for high signal-to-noise ratio galaxies, but may be considerably different at low
    signal-to-noise ratio.)

    Note that ChromaticRealGalaxy objects use arcsec for the units of their linear dimension.  If
    you are using a different unit for other things (the PSF, WCS, etc.), then you should dilate the
    resulting object with `gal.dilate(galsim.arcsec / scale_unit)`.

    Noise from the original images is propagated by this class, though certain restrictions apply
    to when and how that noise is made available.  The propagated noise depends on which Bandpass
    the ChromaticRealGalaxy is being imaged through, so the noise is only available after
    the `drawImage(bandpass, ...)` method has been called.  Also, since ChromaticRealGalaxy will
    only produce reasonable images when convolved with a (suitably wide) PSF, the noise attribute is
    attached to the `ChromaticConvolution` (or `ChromaticTransformation` of the
    `ChromaticConvolution`) which holds as one of its convolutants the `ChromaticRealGalaxy`.

    >>> crg = galsim.ChromaticRealGalaxy(...)
    >>> psf = ...
    >>> obj = galsim.Convolve(crg, psf)
    >>> bandpass = galsim.Bandpass(...)
    >>> assert not hasattr(obj, 'noise')
    >>> image = obj.drawImage(bandpass)
    >>> assert hasattr(obj, 'noise')
    >>> noise1 = obj.noise

    Note that the noise attribute is only associated with the most recently used bandpass.  If you
    draw another image of the same object using a different bandpass, the noise object will be
    replaced.

    >>> bandpass2 = galsim.Bandpass(...)
    >>> image2 = obj.drawImage(bandpass2)
    >>> assert noise1 != obj.noise

    @param real_galaxy_catalogs  A list of `RealGalaxyCatalog` objects from which to create
                            `ChromaticRealGalaxy`s.  Each catalog should represent the same set of
                            galaxies, and in the same order, just imaged through different filters.
                            Note that the number of catalogs must be equal to or larger than the
                            number of SEDs.
    @param index            Index of the desired galaxy in the catalog. [One of `index`, `id`, or
                            `random` is required.]
    @param id               Object ID for the desired galaxy in the catalog. [One of `index`, `id`,
                            or `random` is required.]
    @param random           If True, then just select a completely random galaxy from the catalog.
                            [One of `index`, `id`, or `random` is required.]
    @param rng              A random number generator to use for selecting a random galaxy (may be
                            any kind of BaseDeviate or None) and to use in generating any noise
                            field when padding.  This user-input random number generator takes
                            precedence over any stored within a user-input CorrelatedNoise instance
                            (see `noise_pad` parameter below).  [default: None]
    @param SEDs             An optional list of `SED`s to use when representing real galaxies as
                            sums of separable profiles.  By default, len(real_galaxy_catalogs) SEDs
                            that are polynomials in wavelength will be used.  Note that the number
                            of SEDs must be equal to or smaller than the number of catalogs.
                            [default: see above]
    @param k_interpolant    Either an Interpolant instance or a string indicating which k-space
                            interpolant should be used.  Options are 'nearest', 'sinc', 'linear',
                            'cubic', 'quintic', or 'lanczosN' where N should be the integer order
                            to use.  We strongly recommend leaving this parameter at its default
                            value; see text above for details.  [default: galsim.Quintic()]
    @param maxk             Optional maxk argument.  If you know you will be convolving the
                            resulting `ChromaticRealGalaxy` with a "fat" PSF in a subsequent step,
                            then it can be more efficient to limit the range of Fourier modes used
                            when solving for the sum of separable profiles below.  [default: None]
    @param pad_factor       Factor by which to internally oversample the Fourier-space images
                            that represent the ChromaticRealGalaxy (equivalent to zero-padding the
                            real-space profiles).  We strongly recommend leaving this parameter
                            at its default value; see text in Realgalaxy docstring for details.
                            [default: 4]
    @param normalize_area   By default, the flux of the returned object is normalized such that
                            drawing with exptime=1 and area=1 (the `drawImage` defaults) simulates
                            an image with the appropriate number of counts for a 1 second HST
                            exposure.  Setting this keyword to True will instead normalize the
                            returned object such that to simulate a 1 second HST exposure, you must
                            use drawImage with exptime=1 and area=45238.93416 (the HST collecting
                            area in cm^2).  [default: False]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]
    @param logger           A logger object for output of progress statements if the user wants
                            them.  [default: None]
    Methods
    -------

    There are no additional methods for ChromaticRealGalaxy beyond the usual ChromaticObject
    methods.
    """
    def __init__(self, real_galaxy_catalogs=None, index=None, id=None, random=False, rng=None,
                 gsparams=None, logger=None, **kwargs):
        if rng is None:
            self.rng = galsim.BaseDeviate()
        elif not isinstance(rng, galsim.BaseDeviate):
            raise TypeError("The rng provided to ChromaticRealGalaxy constructor "
                            "is not a BaseDeviate")
        else:
            self.rng = rng
        self._rng = self.rng.duplicate()  # This is only needed if we want to make sure eval(repr)
                                          # results in the same object.

        if real_galaxy_catalogs is None:
            raise ValueError("No RealGalaxyCatalog(s) specified!")

        # Get the index to use in the catalog
        if index is not None:
            if id is not None or random:
                raise AttributeError('Too many methods for selecting a galaxy!')
            use_index = index
        elif id is not None:
            if random:
                raise AttributeError('Too many methods for selecting a galaxy!')
            use_index = real_galaxy_catalogs[0].getIndexForID(id)
        elif random:
            uniform_deviate = galsim.UniformDeviate(self.rng)
            use_index = int(real_galaxy_catalogs[0].nobjects * uniform_deviate())
        else:
            raise AttributeError('No method specified for selecting a galaxy!')
        if logger:
            logger.debug('ChromaticRealGalaxy %d: Start ChromaticRealGalaxy constructor.',
                         use_index)
        self.index = use_index

        # Read in the galaxy, PSF images; for now, rely on pyfits to make I/O errors.
        imgs = [rgc.getGalImage(use_index) for rgc in real_galaxy_catalogs]
        if logger:
            logger.debug('ChromaticRealGalaxy %d: Got gal_image', use_index)

        PSFs = [rgc.getPSF(use_index) for rgc in real_galaxy_catalogs]
        if logger:
            logger.debug('ChromaticRealGalaxy %d: Got psf', use_index)

        bands = [rgc.getBandpass() for rgc in real_galaxy_catalogs]

        xis = []
        for rgc in real_galaxy_catalogs:
            noise_image, pixel_scale, var = rgc.getNoiseProperties(use_index)
            # Make sure xi image is odd-sized.
            if noise_image.array.shape[0] % 2 == 0: #pragma: no branch
                bds = noise_image.bounds
                new_bds = galsim.BoundsI(bds.xmin+1, bds.xmax, bds.ymin+1, bds.ymax)
                noise_image = noise_image[new_bds]
            ii = galsim.InterpolatedImage(noise_image, normalization='sb',
                                          calculate_stepk=False, calculate_maxk=False,
                                          x_interpolant='linear', gsparams=gsparams)
            xi = galsim.correlatednoise._BaseCorrelatedNoise(self.rng, ii, noise_image.wcs)
            xi = xi.withVariance(var)
            xis.append(xi)
        if logger:
            logger.debug('ChromaticRealGalaxy %d: Got noise_image',use_index)
        self.catalog_files = [rgc.getFileName() for rgc in real_galaxy_catalogs]

        self._initialize(imgs, bands, xis, PSFs, gsparams=gsparams, **kwargs)

    @classmethod
    def makeFromImages(cls, images, bands, PSFs, xis, **kwargs):
        """Create a ChromaticRealGalaxy directly from images, bandpasses, PSFs, and noise
        descriptions.  See the ChromaticRealGalaxy docstring for more information.

        @param images           An iterable of high resolution `Images` of a galaxy through
                                different bandpasses.
        @param bands            An iterable of `Bandpass`es corresponding to the input images.
        @param PSFs             Either an iterable of `GSObject`s or `ChromaticObject`s indicating
                                the PSFs of the different input images, or potentially a single
                                `GSObject` or `ChromaticObject` that will be used as the PSF for
                                all images.
        @param xis              An iterable of either `CorrelatedNoise` or `UncorrelatedNoise`
                                objects characterizing the noise in the input images.
        @param SEDs             An optional list of `SED`s to use when representing real galaxies
                                as sums of separable profiles.  By default, len(images) SEDs that
                                are polynomials in wavelength will be used.  Note that the number
                                of SEDs must be equal to or smaller than the number of catalogs.
                                [default: see above]
        @param k_interpolant    Either an Interpolant instance or a string indicating which k-space
                                interpolant should be used.  Options are 'nearest', 'sinc',
                                'linear', 'cubic', 'quintic', or 'lanczosN' where N should be the
                                integer order to use.  We strongly recommend leaving this parameter
                                at its default value; see text above for details.  [default:
                                galsim.Quintic()]
        @param maxk             Optional maxk argument.  If you know you will be convolving the
                                resulting `ChromaticRealGalaxy` with a "fat" PSF in a subsequent
                                step, then it can be more efficient to limit the range of Fourier
                                modes used when solving for the sum of separable profiles below.
                                [default: None]
        @param pad_factor       Factor by which to internally oversample the Fourier-space images
                                that represent the ChromaticRealGalaxy (equivalent to zero-padding
                                the real-space profiles).  We strongly recommend leaving this
                                parameter at its default value; see text in Realgalaxy docstring
                                for details.  [default: 4]
        @param normalize_area   By default, the flux of the returned object is normalized such that
                                drawing with exptime=1 and area=1 (the `drawImage` defaults)
                                simulates an image with the appropriate number of counts for a 1
                                second HST exposure.  Setting this keyword to True will instead
                                normalize the returned object such that to simulate a 1 second HST
                                exposure, you must use drawImage with exptime=1 and
                                area=45238.93416 (the HST collecting area in cm^2).
                                [default: False]
        @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                                details. [default: None]
        @param logger           A logger object for output of progress statements if the user wants
                                them.  [default: None]

        """
        if not hasattr(PSFs, '__iter__'):
            PSFs = [PSFs]*len(images)
        obj = cls.__new__(cls)
        obj._initialize(images, bands, xis, PSFs, **kwargs)
        return obj

    def _initialize(self, imgs, bands, xis, PSFs,
                    SEDs=None, k_interpolant=None, maxk=None, pad_factor=4., normalize_area=False,
                    gsparams=None):
        if SEDs is None:
            SEDs = self._poly_SEDs(bands)
        self.SEDs = SEDs

        self._normalize_area = normalize_area
        self._k_interpolant = k_interpolant
        self._gsparams = gsparams

        NSED = len(self.SEDs)
        Nim = len(imgs)
        assert Nim == len(bands)
        assert Nim == len(xis)
        assert Nim == len(PSFs)
        assert Nim >= NSED

        if normalize_area:
            imgs = [img/HST_area for img in imgs]
            xis = [xi/HST_area/HST_area for xi in xis]

        # Need to sample three different types of objects on the same Fourier grid: the input
        # effective PSFs, the input images, and the input correlation-functions/power-spectra.
        # There are quite a few potential options for implementing this Fourier sampling.  Some
        # examples include:
        #   * draw object in real space, interpolate onto the real-space grid conjugate to the
        #     desired Fourier-space grid and then DFT with numpy.fft methods.
        #   * Use numpy.fft methods on pre-sampled real-space input (like the input images), then
        #     use an InterpolatedKImage object to regrid onto desired Fourier grid.
        #   * Create an InterpolatedImage from pre-sampled input then use drawKImage to directly
        #     sample on desired Fourier grid.
        # I'm sure there are other options too.  The options chosen below were chosen empirically
        # based on tests of propagating both (chromatic) galaxy images and images of pure noise.

        # Select maxk by requiring modes to be resolved both by the marginal PSFs (i.e., the
        # achromatic PSFs obtained by evaluating the chromatic PSF at the blue and red edges of
        # each of the filters provided) and also by the input images' pixel scales.

        img_maxk = np.min([np.pi/img.scale for img in imgs])
        marginal_PSFs = [PSF.evaluateAtWavelength(band.blue_limit)
                         for PSF in PSFs for band in bands]
        marginal_PSFs += [PSF.evaluateAtWavelength(band.red_limit)
                          for PSF in PSFs for band in bands]
        psf_maxk = np.min([p.maxK() for p in marginal_PSFs])

        # In practice, the output PSF should almost always cut off at smaller maxk than obtained
        # above.  In this case, the user can set the maxK keyword argument for improved efficiency.
        if maxk is None:
            maxk = np.min([img_maxk, psf_maxk])
        else:
            maxk = np.min([img_maxk, psf_maxk, maxk])

        # Setting stepk is trickier.  We'll assume that the postage stamp inputs are already at the
        # critical size to avoid significant aliasing and use the implied stepk.  We'll insist that
        # the WCS is a simple PixelScale.  We'll also use the same trick that InterpolatedImage
        # uses to improve accuracy, namely, increase the Fourier-space resolution a factor of
        # `pad_factor`.
        stepk = np.min([2*np.pi/(img.scale*max(img.array.shape))/pad_factor for img in imgs])
        nk = 2*int(np.floor(maxk/stepk))

        # Create Fourier-space kimages of effective PSFs
        PSF_eff_kimgs = np.empty((Nim, NSED, nk, nk), dtype=np.complex128)
        for i, (img, band, PSF) in enumerate(zip(imgs, bands, PSFs)):
            for j, sed in enumerate(self.SEDs):
                # assume that PSF already includes pixel, so don't convolve one in again.
                PSF_eff_kimgs[i, j] = (PSF * sed).drawKImage(band, nx=nk, ny=nk, scale=stepk).array

        # Get Fourier-space representations of input imgs.
        kimgs = np.empty((Nim, nk, nk), dtype=np.complex128)

        for i, img in enumerate(imgs):
            ii = galsim.InterpolatedImage(img)
            kimgs[i] = ii.drawKImage(nx=nk, ny=nk, scale=stepk).array

        # Setup input noise power spectra
        pks = np.empty((Nim, nk, nk), dtype=np.float64)
        for i, (img, xi) in enumerate(zip(imgs, xis)):
            pks[i] = xi.drawKImage(nx=nk, ny=nk, scale=stepk).array.real / xi.wcs.pixelArea()
            ny, nx = img.array.shape
            pks[i] *= nx * ny
        w = 1./np.sqrt(pks)

        # Allocate and fill output coefficients and covariances.
        # Note: put NSED axis last, since significantly faster to compute them this way,
        # even though we eventually convert to images which are strided in this format.
        coef = np.zeros((nk, nk, NSED), dtype=np.complex128)
        Sigma = np.empty((nk, nk, NSED, NSED), dtype=np.complex128)

        # Solve the weighted linear least squares problem for each Fourier mode.  This is
        # effectively a constrained chromatic deconvolution.  Take advantage of symmetries.
        galsim._galsim.ComputeCRGCoefficients(
            coef.ctypes.data, Sigma.ctypes.data,
            w.ctypes.data, kimgs.ctypes.data, PSF_eff_kimgs.ctypes.data,
            NSED, Nim, nk, nk)

        # Reorder these so they correspond to (NSED, nky, nkx) and (NSED, NSED, nky, nkx) shapes.
        coef = np.moveaxis(coef, (0,1,2), (1,2,0))
        Sigma = np.moveaxis(Sigma, (0,1,2,3), (2,3,0,1))

        # Set up objlist as required of ChromaticSum subclass.
        objlist = []
        for i, sed in enumerate(self.SEDs):
            objlist.append(sed * galsim.InterpolatedKImage(galsim.ImageCD(coef[i], scale=stepk)))

        Sigma_dict = {}
        for i in range(NSED):
            for j in range(i, NSED):
                obj = galsim.InterpolatedKImage(galsim.ImageCD(Sigma[i, j], scale=stepk))
                obj /= (imgs[0].array.shape[0] * imgs[0].array.shape[1] * imgs[0].scale**2)
                Sigma_dict[(i, j)] = obj

        self.covspec = galsim.CovarianceSpectrum(Sigma_dict, self.SEDs)

        ChromaticSum.__init__(self, objlist)

    @staticmethod
    def _poly_SEDs(bands):
        # Use polynomial SEDs by default; up to the number of bands provided.
        waves = []
        for bp in bands:
            waves = np.union1d(waves, bp.wave_list)
        SEDs = []
        for i in range(len(bands)):
            SEDs.append(
                    galsim.SED(galsim.LookupTable(waves, waves**i, interpolant='linear'),
                               'nm', 'fphotons')
                    .withFlux(1.0, bands[0]))
        return SEDs

    def __eq__(self, other):
        return (isinstance(other, galsim.ChromaticRealGalaxy) and
                self.catalog_files == other.catalog_files and
                self.index == other.index and
                self.SEDs == other.SEDs and
                self._k_interpolant == other._k_interpolant and
                self._rng == other._rng and
                self._normalize_area == other._normalize_area and
                self._gsparams == other._gsparams)
    def __ne__(self, other): return not self.__eq__(other)

    def __hash__(self):
        return hash(("galsim.ChromaticRealGalaxy", tuple(self.catalog_files), self.index,
                     tuple(self.SEDs), self._k_interpolant, self._rng.serialize(),
                     self._normalize_area, self._gsparams))

    def __str__(self):
        return "galsim.ChromaticRealGalaxy(%r, index=%r)"%(self.catalog_files, self.index)

    def __repr__(self):
        return ("galsim.ChromaticRealGalaxy(%r, SEDs=%r, index=%r, rng=%r, k_interpolant=%r, "
                "normalize_area=%r, gsparams=%r)"%(self.catalog_files, self.SEDs, self.index,
                                                   self._rng, self._k_interpolant,
                                                   self._normalize_area, self._gsparams))
