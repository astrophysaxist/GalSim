# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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
# list of conditions, and the disclaimer given in the accompanying LICENSE file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions, and the disclaimer given in the documentation
# and/or other materials provided with the distribution.
#
"""@file detectors.py

Module with routines to simulate CCD and NIR detector effects like nonlinearity, reciprocity
failure, interpixel capacitance, etc.
"""

import galsim
import numpy
import sys

def applyNonlinearity(self, NLfunc, *args):
    """
    Applies the given non-linearity function (`NLfunc`) on the Image instance directly.

    This routine can transform the image in a non-linear manner specified by the user. However,
    the typical kind of non-linearity one sees in astronomical images is voltage non-linearity,
    also sometimes known as 'classical non-linearity', refers to the non-linearity in
    charge-to-voltage conversion process. This arises as charge gets integrated at the junction
    capacitance of the pixel node. Voltage non-linearity decreases signals at higher signal
    levels, causing the attenuation of brighter pixels. The image should include both the
    signal from the astronomical objects as well as the background level. Other detectors effects
    such as dark current and persistence (not currently included in GalSim) would also occur
    before the inclusion of nonlinearity.

    The argument `NLfunc` is a callable function (for example a lambda function, a
    galsim.LookupTable, or a user-defined function), possibly with arguments that need to be given
    as subsequent arguments to the `applyNonlinearity` function (after the `NLfunc` argument).
    `NLfunc` should be able to take a 2d NumPy array as input, and return a NumPy array of the same
    shape.  It should be defined such that it outputs the final image with nonlinearity included
    (i.e., in the limit that there is no nonlinearity, the function should return the original
    image, NOT zero). The image should be in units of electrons when this routine is being used
    to generate classical non-linearity. When used for other purposes, the units can be in
    electrons or in ADU, as found appropriate by the user.

    Calling with no parameter:
    -------------------------

        >>> f = lambda x: x + (1.e-7)*(x**2)
        >>> img.applyNonlinearity(f)

    Calling with 1 or more parameters:
    ---------------------------------

        >>> f = lambda x, beta1, beta2: x - beta1*x*x + beta2*x*x*x
        >>> img.applyNonlinearity(f, 1.e-7, 1.e-10)

    On calling the method, the Image instance `img` is transformed by the user-defined function `f`
    with `beta1` = 1.e-7 and `beta2` = 1.e-10.

    @param NLfunc    The function that maps the input image pixel values to the output image pixel
                     values. 
    @param *args     Any subsequent arguments are passed along to the NLfunc function.

    """

    # Extract out the array from Image since not all functions can act directly on Images
    result = NLfunc(self.array,*args)
    if not isinstance(result, numpy.ndarray):
        raise ValueError("NLfunc does not return a NumPy array.")
    if self.array.shape != result.shape:
        raise ValueError("NLfunc does not return a NumPy array of the same shape as input!")
    self.array[:,:] = result


def addReciprocityFailure(self, exp_time, alpha, base_flux):
    """
    Accounts for the reciprocity failure and corrects the original Image for it directly.

    Reciprocity, in the context of photography, is the inverse relationship between the incident
    flux (I) of a source object and the exposure time (t) required to produce a given response (p)
    in the detector, i.e., p = I*t. At very low (also at high) levels of incident flux, deviation
    from this relation is observed, leading to reduced sensitivity at low flux levels. The pixel
    response to a high flux is larger than its response to a low flux. This flux-dependent non-
    linearity is known as 'Reciprocity Failure' and is known to happen in photographic films since
    1893. Interested users can refer to http://en.wikipedia.org/wiki/Reciprocity_(photography)

    CCDs are not known to suffer from this effect. HgCdTe detectors that are used for near infrared
    astrometry, although to an extent much lesser than the photographic films, are found to
    exhibit reciprocity failure at low flux levels. The exact mechanism of this effect is unknown
    and hence we lack a good theoretical model. Many models that fit the empirical data exist and
    a common relation is

            pR/p = (1 + alpha*log10(p/t) - alpha*log10(p'/t'))

    where T is the exposure time (in units of seconds), p is the pixel response (in units of
    electrons) and pR is the response if the reciprocity relation were to hold. p'/T' is count
    rate (in electrons/second) corresponding to the photon flux (base flux) at which the detector
    is calibrated to have its nominal gain. alpha is the parameter in the model, measured in units
    of per decade and varies with detectors and the operating temperature. The functional form for
    the reciprocity failure is motivated empirically from the tests carried out on H2RG detectors.
    See for reference Fig. 1 and Fig. 2 of http://arxiv.org/abs/1106.1090. Since pR/p remains
    close to unity over a wide range of flux, we convert this relation to a power law by
    approximating (pR/p)-1 ~ log(pR/p). This gives a relation that is better behaved than the
    logarithmic relation at low flux levels.

            pR/p = ((p/t)/(p'/t'))^(alpha/log(10)).

    Because of how this function is defined, the input image must have non-negative pixel
    values for the resulting image to be well-defined. Negative pixel values result in 'nan's.
    The image should be in units of electrons, or if it is in ADU, then the value passed to
    exp_time should be the exposure time divided by the nominal gain. The image should include
    both the signal from the astronomical objects as well as the background level.  The addition of
    nonlinearity should occur after including the effect of reciprocity failure.

    Calling
    -------

        >>>  img.addReciprocityFailure(exp_time, alpha, base_flux)

    @param exp_time         The exposure time (t) in seconds, which goes into the expression for
                            reciprocity failure given in the docstring.
    @param alpha            The alpha parameter in the expression for reciprocity failure, in
                            units of 'per decade'.
    @param base_flux        The flux (p'/t') at which the gain is calibrated to have its nominal
                            value.
    
    @returns None
    """

    if not isinstance(alpha, float) or alpha < 0.:
        raise ValueError("Invalid value of alpha, must be float >= 0")
    if not (isinstance(exp_time, float) or isinstance(exp_time, int)) or exp_time < 0.:
        raise ValueError("Invalid value of exp_time, must be float or int >= 0")
    if not (isinstance(base_flux, float) or isinstance(base_flux,int)) or base_flux < 0.:
        raise ValueError("Invalid value of base_flux, must be float or int >= 0")

    if numpy.any(self.array<0):
        import warnings
        warnings.warn("One or more pixel values are negative and will be set as 'nan'.")

    p0 = exp_time*base_flux
    a = alpha/numpy.log(10)
    self.applyNonlinearity(lambda x,x0,a: (x**(a+1))/(x0**a), p0, a)

def applyIPC(self, IPC_kernel, edge_treatment='extend', fill_value=None, kernel_nonnegativity=True,
    kernel_normalization=True):
    """
    Applies the effect of interpixel capacitance to the Image instance.

    In NIR detectors, the quantity that is sensed is not the charge as in CCDs, but a voltage that
    relates to the charge present within each pixel. The voltage read at a given pixel location is
    influenced by the charges present in the neighboring pixel locations due to capacitive
    coupling of sense nodes.

    This interpixel capacitance is approximated as a linear effect that can be described by a 3x3
    kernel that is convolved with the image. The kernel must be an Image instance and could be
    intrinsically anisotropic. A sensible kernel must have non-negative entries and must be
    normalized such that the sum of the elements is 1, in order to conserve the total charge.
    The (1,1) element of the kernel is the contribution to the voltage read at a pixel from the
    electrons in the pixel to its bottom-left, the (1,2) element of the kernel is the contribution
    from the charges to its left and so on.

    The argument 'edge_treatment' specifies how the edges of the image should be treated, which
    could be in one of the three ways:
    
    1. 'extend': The kernel is convolved with the zero-padded image, leading to a larger
        intermediate image. The central portion of this image is returned.  [default]
    2. 'crop': The kernel is convolved with the image, with the kernel inside the image completely.
        Pixels at the edges, where the center of the kernel could not be placed, are set to the
        value specified by 'fill_value'. If 'fill_value' is not specified or set to 'None', then
        the pixel values in the original image are retained. The user can make the edges invalid
        by setting fill_value to numpy.nan.
    3. 'wrap': The kernel is convolved with the image, assuming periodic boundary conditions.

    The size of the image array remains unchanged in all three cases.

    Calling
    -------

        >>> img.applyIPC(IPC_kernel=ipc_kernel, edge_treatment='extend', fill_value=0,
            kernel_nonnegativity=True, kernel_normalization=True)

    @param IPC_kernel              A 3x3 Image instance that is convolved with the Image instance
    @param edge_treatment          Specifies the method of handling edges and should be one of
                                   'crop', 'extend' or 'wrap'. See above for details.
                                   [default: 'extend']
    @param fill_value              Specifies the value (including nan) to fill the edges with when
                                   edge_treatment is 'crop'. If unspecified or set to 'None', the
                                   original pixel values are retained at the edges. If
                                   edge_treatment is not 'crop', then this is ignored.
    @param kernel_nonnegativity    Specify whether the kernel should have only non-negative
                                   entries.  [default: True]
    @param kernel_normalization    Specify whether to check and enforce correct normalization for
                                   the kernel.  [default: True]

    @returns None
    """

    # IPC kernel has to be a 3x3 Image instance
    if not isinstance(IPC_kernel,galsim.Image):
        raise ValueError("IPC_kernel must be an Image instance .")
    ipc_kernel = IPC_kernel.array
    if not ipc_kernel.shape==(3,3):
        raise ValueError("IPC kernel must be an Image instance of size 3x3.")

    # Check for non-negativity of the kernel
    if kernel_nonnegativity is True:
        if (ipc_kernel<0).any() is True:
            raise ValueError("IPC kernel must not contain negative entries")

    # Check and enforce correct normalization for the kernel
    if kernel_normalization is True:
        if abs(ipc_kernel.sum() - 1.0) > 10.*numpy.finfo(ipc_kernel.dtype.type).eps:
            import warnings
            warnings.warn("The entries in the IPC kernel did not sum to 1. Scaling the kernel to "\
                +"ensure correct normalization.")
            IPC_kernel = IPC_kernel/ipc_kernel.sum()

    # edge_treatment can be 'extend', 'wrap' or 'crop'
    if edge_treatment=='crop':
        # Simply re-label the array of the Image instance
        pad_array = self.array
    elif edge_treatment=='extend':
        # Copy the array of the Image instance and pad with zeros
        pad_array = numpy.zeros((self.array.shape[0]+2,self.array.shape[1]+2))
        pad_array[1:-1,1:-1] = self.array
    elif edge_treatment=='wrap':
        # Copy the array of the Image instance and pad with zeros initially
        pad_array = numpy.zeros((self.array.shape[0]+2,self.array.shape[1]+2))
        pad_array[1:-1,1:-1] = self.array
        # and wrap around the edges
        pad_array[0,:] = pad_array[-2,:]
        pad_array[-1,:] = pad_array[1,:]
        pad_array[:,0] = pad_array[:,-2]
        pad_array[:,-1] = pad_array[:,1]
    else:
        raise ValueError("edge_treatment has to be one of 'extend', 'wrap' or 'crop'. ")

    # Generating different segments of the padded array
    center = pad_array[1:-1,1:-1]
    top = pad_array[2:,1:-1]
    bottom = pad_array[:-2,1:-1]
    left = pad_array[1:-1,:-2]
    right = pad_array[1:-1,2:]
    topleft = pad_array[2:,:-2]
    bottomright = pad_array[:-2,2:]
    topright = pad_array[2:,2:]
    bottomleft = pad_array[:-2,:-2]

    # Ensure that the choice of origin does not matter
    x0 = IPC_kernel.bounds.xmin # 1 by default
    y0 = IPC_kernel.bounds.ymin # 1 by default

    # Generating the output array, with 2 rows and 2 columns lesser than the padded array
    # Image values have been used to make the code look more intuitive
    out_array = \
        IPC_kernel(x0,y0+2)*topleft + IPC_kernel(x0+1,y0+2)*top + IPC_kernel(x0+2,y0+2)*topright +\
        IPC_kernel(x0,y0+1)*left + IPC_kernel(x0+1,y0+1)*center + IPC_kernel(x0+2,y0+1)*right +\
        IPC_kernel(x0,y0)*bottomleft + IPC_kernel(x0+1,y0)*bottom + IPC_kernel(x0+2,y0)*bottomright

    if edge_treatment=='crop':
        self.array[1:-1,1:-1] = out_array
        #Explicit edge effects handling with filling the edges with the value given in fill_value
        if fill_value is not None:
            if isinstance(fill_value, float) or isinstance(fill_value, int):
                self.array[0,:] = fill_value
                self.array[-1,:] = fill_value
                self.array[:,0] = fill_value
                self.array[:,-1] = fill_value
            else:
                raise ValueError("'fill_value' must be either a float or an int")
    else:
        self.array[:,:] = out_array

galsim.Image.applyNonlinearity = applyNonlinearity
galsim.Image.addReciprocityFailure = addReciprocityFailure
galsim.Image.applyIPC = applyIPC
