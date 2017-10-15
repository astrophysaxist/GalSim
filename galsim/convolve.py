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
"""@file convolve.py
Some compound GSObject classes that contain other GSObject instances:

Convolution = convolution of multiple profiles
Deconvolution = deconvolution by a given profile
AutoConvolution = convolution of a profile by itself
AutoCorrelation = convolution of a profile by its reflection
"""

import numpy as np

from . import _galsim
from .gsparams import GSParams
from .gsobject import GSObject
from .chromatic import ChromaticObject, ChromaticConvolution
from .utilities import lazy_property


def Convolve(*args, **kwargs):
    """A function for convolving 2 or more GSObject or ChromaticObject instances.

    This function will inspect its input arguments to decide if a Convolution object or a
    ChromaticConvolution object is required to represent the convolution of surface
    brightness profiles.

    @param args             Unnamed args should be a list of objects to convolve.
    @param real_space       Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the objects have hard
                            edges.]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    @returns a Convolution or ChromaticConvolution instance as appropriate.
    """
    from .chromatic import ChromaticConvolution
    # First check for number of arguments != 0
    if len(args) == 0:
        raise TypeError("At least one ChromaticObject or GSObject must be provided.")
    elif len(args) == 1:
        if isinstance(args[0], (GSObject, ChromaticObject)):
            args = [args[0]]
        elif isinstance(args[0], list) or isinstance(args[0], tuple):
            args = args[0]
        else:
            raise TypeError("Single input argument must be a GSObject, ChromaticObject, "
                            + "or a (possibly mixed) list of them.")
    # else args is already the list of objects

    if any([isinstance(a, ChromaticObject) for a in args]):
        return ChromaticConvolution(*args, **kwargs)
    else:
        return Convolution(*args, **kwargs)


class Convolution(GSObject):
    """A class for convolving 2 or more GSObject instances.

    The convolution will normally be done using discrete Fourier transforms of each of the component
    profiles, multiplying them together, and then transforming back to real space.

    There is also an option to do the convolution as integrals in real space.  To do this, use the
    optional keyword argument `real_space = True`.  Currently, the real-space integration is only
    enabled for convolving 2 profiles.  (Aside from the trivial implementaion for 1 profile.)  If
    you try to use it for more than 2 profiles, an exception will be raised.

    The real-space convolution is normally slower than the DFT convolution.  The exception is if
    both component profiles have hard edges, e.g. a truncated Moffat or Sersic with a Pixel.  In
    that case, the highest frequency `maxk` for each component is quite large since the ringing dies
    off fairly slowly.  So it can be quicker to use real-space convolution instead.  Also,
    real-space convolution tends to be more accurate in this case as well.

    If you do not specify either `real_space = True` or `False` explicitly, then we check if there
    are 2 profiles, both of which have hard edges.  In this case, we automatically use real-space
    convolution.  In all other cases, the default is not to use real-space convolution.

    Initialization
    --------------

    The normal way to use this class is to use the Convolve() factory function:

        >>> gal = galsim.Sersic(n, half_light_radius)
        >>> psf = galsim.Gaussian(sigma)
        >>> final = galsim.Convolve([gal, psf])

    The objects to be convolved may be provided either as multiple unnamed arguments (e.g.
    `Convolve(psf, gal)`) or as a list (e.g. `Convolve([psf, gal])`).  Any number of objects may
    be provided using either syntax.  (Well, the list has to include at least 1 item.)

    @param args             Unnamed args should be a list of objects to convolve.
    @param real_space       Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the objects have hard
                            edges.]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Note: if `gsparams` is unspecified (or None), then the Convolution instance inherits the same
    GSParams as the first item in the list.  Also, note that parameters related to the Fourier-
    space calculations must be set when initializing the individual GSObjects that go into the
    Convolution, NOT when creating the Convolution (at which point the accuracy and threshold
    parameters will simply be ignored).

    Methods
    -------

    There are no additional methods for Convolution beyond the usual GSObject methods.
    """
    def __init__(self, *args, **kwargs):
        # First check for number of arguments != 0
        if len(args) == 0:
            raise TypeError("At least one ChromaticObject or GSObject must be provided.")
        elif len(args) == 1:
            if isinstance(args[0], GSObject):
                args = [args[0]]
            elif isinstance(args[0], list) or isinstance(args[0], tuple):
                args = args[0]
            else:
                raise TypeError("Single input argument must be a GSObject or list of them.")
        # else args is already the list of objects

        # Check kwargs
        # real_space can be True or False (default if omitted is None), which specifies whether to
        # do the convolution as an integral in real space rather than as a product in fourier
        # space.  If the parameter is omitted (or explicitly given as None I guess), then
        # we will usually do the fourier method.  However, if there are 2 components _and_ both of
        # them have hard edges, then we use real-space convolution.
        real_space = kwargs.pop("real_space", None)
        gsparams = kwargs.pop("gsparams", None)

        # Make sure there is nothing left in the dict.
        if kwargs:
            raise TypeError(
                "Convolution constructor got unexpected keyword argument(s): %s"%kwargs.keys())

        # Check whether to perform real space convolution...
        # Start by checking if all objects have a hard edge.
        hard_edge = True
        for obj in args:
            if not isinstance(obj, GSObject):
                raise TypeError("Arguments to Convolution must be GSObjects, not %s"%obj)
            if not obj.has_hard_edges:
                hard_edge = False

        if real_space is None:
            # The automatic determination is to use real_space if 2 items, both with hard edges.
            if len(args) <= 2:
                real_space = hard_edge
            else:
                real_space = False
        elif bool(real_space) != real_space:
            raise TypeError("real_space must be a boolean")

        # Warn if doing DFT convolution for objects with hard edges
        if not real_space and hard_edge:

            import warnings
            if len(args) == 2:
                msg = """
                Doing convolution of 2 objects, both with hard edges.
                This might be more accurate and/or faster using real_space=True"""
            else:
                msg = """
                Doing convolution where all objects have hard edges.
                There might be some inaccuracies due to ringing in k-space."""
            warnings.warn(msg)

        if real_space:
            # Can't do real space if nobj > 2
            if len(args) > 2:
                import warnings
                msg = """
                Real-space convolution of more than 2 objects is not implemented.
                Switching to DFT method."""
                warnings.warn(msg)
                real_space = False

            # Also can't do real space if any object is not analytic, so check for that.
            else:
                for obj in args:
                    if not obj.is_analytic_x:
                        import warnings
                        msg = """
                        A component to be convolved is not analytic in real space.
                        Cannot use real space convolution.
                        Switching to DFT method."""
                        warnings.warn(msg)
                        real_space = False
                        break

        # Save the construction parameters (as they are at this point) as attributes so they
        # can be inspected later if necessary.
        self._real_space = bool(real_space)
        self._obj_list = args
        self._gsparams = GSParams.check(gsparams, self._obj_list[0].gsparams)

    @property
    def obj_list(self): return self._obj_list

    @property
    def real_space(self): return self._real_space

    @lazy_property
    def _sbp(self):
        SBList = [obj._sbp for obj in self.obj_list]
        return _galsim.SBConvolve(SBList, self._real_space, self.gsparams._gsp)

    @lazy_property
    def _noise(self):
        # If one of the objects has a noise attribute, then we convolve it by the others.
        # More than one is not allowed.
        _noise = None
        for i, obj in enumerate(self.obj_list):
            if obj.noise is not None:
                if _noise is not None:
                    import warnings
                    warnings.warn("Unable to propagate noise in galsim.Convolution when "
                                  "multiple objects have noise attribute")
                    break
                _noise = obj.noise
                others = [ obj2 for k, obj2 in enumerate(self.obj_list) if k != i ]
                assert len(others) > 0
                if len(others) == 1:
                    _noise = _noise.convolvedWith(others[0])
                else:
                    _noise = _noise.convolvedWith(Convolve(others))
        return _noise

    def __eq__(self, other):
        return (isinstance(other, Convolution) and
                self.obj_list == other.obj_list and
                self.real_space == other.real_space and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.Convolution", tuple(self.obj_list), self.real_space, self.gsparams))

    def __repr__(self):
        return 'galsim.Convolution(%r, real_space=%r, gsparams=%r)'%(
                self.obj_list, self.real_space, self.gsparams)

    def __str__(self):
        str_list = [ str(obj) for obj in self.obj_list ]
        s = 'galsim.Convolve(%s'%(', '.join(str_list))
        if self.real_space:
            s += ', real_space=True'
        s += ')'
        return s

    def _prepareDraw(self):
        for obj in self.obj_list:
            obj._prepareDraw()

    @property
    def _maxk(self):
        maxk_list = [obj.maxk for obj in self.obj_list]
        return np.min(maxk_list)

    @property
    def _stepk(self):
        # This is approximate.  stepk ~ 2pi/R
        # Assume R_final^2 = Sum(R_i^2)
        # So 1/stepk^2 = 1/Sum(1/stepk_i^2)
        inv_stepksq_list = [obj.stepk**(-2) for obj in self.obj_list]
        return np.sum(inv_stepksq_list)**(-0.5)

    @property
    def _has_hard_edges(self):
        return len(self.obj_list) == 1 and self.obj_list[0].has_hard_edges

    @property
    def _is_axisymmetric(self):
        axi_list = [obj.is_axisymmetric for obj in self.obj_list]
        return bool(np.all(axi_list))

    @property
    def _is_analytic_x(self):
        if self.real_space and len(self.obj_list) <= 2:
            ax_list = [obj.is_analytic_x for obj in self.obj_list]
            return bool(np.all(ax_list))
        else:
            return False

    @property
    def _is_analytic_k(self):
        ak_list = [obj.is_analytic_k for obj in self.obj_list]
        return bool(np.all(ak_list))

    @property
    def _centroid(self):
        cen_list = [obj.centroid for obj in self.obj_list]
        return sum(cen_list[1:], cen_list[0])

    @lazy_property
    def _flux(self):
        flux_list = [obj.flux for obj in self.obj_list]
        return np.prod(flux_list)

    @property
    def _positive_flux(self):
        pflux_list = [obj.positive_flux for obj in self.obj_list]
        return np.prod(pflux_list)

    @property
    def _negative_flux(self):
        pflux_list = [obj.negative_flux for obj in self.obj_list]
        return np.prod(pflux_list)

    @property
    def _max_sb(self):
        # This one is probably the least accurate of all the estimates of maxSB.
        # The calculation is based on the exact value for Gaussians.
        #     maxSB = flux / 2pi sigma^2
        # When convolving multiple Gaussians together, the sigma^2 values add:
        #     sigma_final^2 = Sum_i sigma_i^2
        # from which we can calculate
        #     maxSB = flux_final / 2pi sigma_final^2
        # or
        #     maxSB = flux_final / Sum_i (flux_i / maxSB_i)
        #
        # For non-Gaussians, this procedure will tend to produce an over-estimate of the
        # true maximum SB.  Non-Gaussian profiles tend to have peakier parts which get smoothed
        # more than the Gaussian does.  So this is likely to be too high, which is acceptable.
        area_list = [obj.flux / obj.max_sb for obj in self.obj_list]
        return self.flux / np.sum(area_list)

    def _xValue(self, pos):
        if len(self.obj_list) == 1:
            return self.obj_list[0]._xValue(pos)
        elif len(self.obj_list) == 2:
            try:
                return self._sbp.xValue(pos._p)
            except AttributeError:
                raise NotImplementedError(
                    "At least one profile in %s does not implement real-space convolution"%self)
        else:
            raise ValueError("Cannot use real_space convolution for >2 profiles")

    def _kValue(self, pos):
        kv_list = [obj.kValue(pos) for obj in self.obj_list]
        return np.prod(kv_list)

    def _drawReal(self, image):
        if len(self.obj_list) == 1:
            self.obj_list[0]._drawReal(image)
        elif len(self.obj_list) == 2:
            try:
                self._sbp.draw(image._image, image.scale)
            except AttributeError:
                raise ValueError("Cannot use real_space convolution for these profiles")
        else:
            raise ValueError("Cannot use real_space convolution for >2 profiles")

    def _shoot(self, photons, ud):
        from .photon_array import PhotonArray

        self.obj_list[0]._shoot(photons, ud)
        # It may be necessary to shuffle when convolving because we do not have a
        # gaurantee that the convolvee's photons are uncorrelated, e.g., they might
        # both have their negative ones at the end.
        # However, this decision is now made by the convolve method.
        for obj in self.obj_list[1:]:
            p1 = PhotonArray(len(photons))
            obj._shoot(p1, ud)
            photons.convolve(p1, ud)

    def _drawKImage(self, image):
        self.obj_list[0]._drawKImage(image)
        if len(self.obj_list) > 1:
            im1 = image.copy()
            for obj in self.obj_list[1:]:
                obj._drawKImage(im1)
                image *= im1

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_sbp',None)
        return d

    def __setstate__(self, d):
        self.__dict__ = d



def Deconvolve(obj, gsparams=None):
    """A function for deconvolving by either a GSObject or ChromaticObject.

    This function will inspect its input argument to decide if a Deconvolution object or a
    ChromaticDeconvolution object is required to represent the deconvolution by a surface
    brightness profile.

    @param obj              The object to deconvolve.
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    @returns a Deconvolution or ChromaticDeconvolution instance as appropriate.
    """
    from .chromatic import ChromaticDeconvolution
    if isinstance(obj, ChromaticObject):
        return ChromaticDeconvolution(obj, gsparams=gsparams)
    elif isinstance(obj, GSObject):
        return Deconvolution(obj, gsparams=gsparams)
    else:
        raise TypeError("Argument to Deconvolve must be either a GSObject or a ChromaticObject.")


class Deconvolution(GSObject):
    """A class for deconvolving a GSObject.

    The Deconvolution class represents a deconvolution kernel.  Note that the Deconvolution class,
    or compound objects (Sum, Convolution) that include a Deconvolution as one of the components,
    cannot be photon-shot using the 'phot' method of drawImage() method.

    You may also specify a `gsparams` argument.  See the docstring for GSParams using
    `help(galsim.GSParams)` for more information about this option.  Note: if `gsparams` is
    unspecified (or None), then the Deconvolution instance inherits the same GSParams as the object
    being deconvolved.

    Initialization
    --------------

    The normal way to use this class is to use the Deconvolve() factory function:

        >>> inv_psf = galsim.Deconvolve(psf)
        >>> deconv_gal = galsim.Convolve(inv_psf, gal)

    @param obj              The object to deconvolve.
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods
    -------

    There are no additional methods for Deconvolution beyond the usual GSObject methods.
    """
    _has_hard_edges = False
    _is_analytic_x = False

    def __init__(self, obj, gsparams=None):
        if not isinstance(obj, GSObject):
            raise TypeError("Argument to Deconvolution must be a GSObject.")

        # Save the original object as an attribute, so it can be inspected later if necessary.
        self._orig_obj = obj
        self._gsparams = GSParams.check(gsparams, self._orig_obj.gsparams)
        self._min_acc_kvalue = obj.flux * self.gsparams.kvalue_accuracy
        self._inv_min_acc_kvalue = 1./self._min_acc_kvalue

    @lazy_property
    def _sbp(self):
        return _galsim.SBDeconvolve(self.orig_obj._sbp, self.gsparams._gsp)

    @property
    def orig_obj(self): return self._orig_obj

    @property
    def _noise(self):
        if self.orig_obj.noise is not None:
            import warnings
            warnings.warn("Unable to propagate noise in galsim.Deconvolution")
        return None

    def __eq__(self, other):
        return (isinstance(other, Deconvolution) and
                self.orig_obj == other.orig_obj and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.Deconvolution", self.orig_obj, self.gsparams))

    def __repr__(self):
        return 'galsim.Deconvolution(%r, gsparams=%r)'%(self.orig_obj, self.gsparams)

    def __str__(self):
        return 'galsim.Deconvolve(%s)'%self.orig_obj

    def _prepareDraw(self):
        self.orig_obj._prepareDraw()

    @property
    def _maxk(self):
        return self.orig_obj.maxk

    @property
    def _stepk(self):
        return self.orig_obj.stepk

    @property
    def _is_axisymmetric(self):
        return self.orig_obj.is_axisymmetric

    @property
    def _is_analytic_k(self):
        return self.orig_obj.is_analytic_k

    @property
    def _centroid(self):
        return -self.orig_obj.centroid

    @lazy_property
    def _flux(self):
        return 1./self.orig_obj.flux

    @property
    def _positive_flux(self):
        return 1./self.orig_obj.positive_flux

    @property
    def _negative_flux(self):
        return 1./self.orig_obj.negative_flux

    @property
    def _max_sb(self):
        # The only way to really give this any meaning is to consider it in the context
        # of being part of a larger convolution with other components.  The calculation
        # of maxSB for Convolve is
        #     maxSB = flux_final / Sum_i (flux_i / maxSB_i)
        #
        # A deconvolution will contribute a -sigma^2 to the sum, so a logical choice for
        # maxSB is to have flux / maxSB = -flux_adaptee / maxSB_adaptee, so its contribution
        # to the Sum_i 2pi sigma^2 is to subtract its adaptee's value of sigma^2.
        #
        # maxSB = -flux * maxSB_adaptee / flux_adaptee
        #       = -maxSB_adaptee / flux_adaptee^2
        #
        return -self.orig_obj.max_sb / self.orig_obj.flux**2

    def _xValue(self, pos):
        raise NotImplementedError("Cannot evaluate a Deconvolution in real space")

    def _kValue(self, pos):
        # Really, for very low original kvalues, this gets very high, which can be unstable
        # in the presence of noise.  So if the original value is less than min_acc_kvalue,
        # we instead just return 1/min_acc_kvalue rather than the real inverse.
        kval = self.orig_obj._kValue(pos)
        if abs(kval) < self._min_acc_kvalue:
            return self._inv_min_acc_kvalue
        else:
            return 1./kval

    def _drawReal(self, image):
        raise NotImplementedError("Cannot draw a Deconvolution in real space")

    def _shoot(self, photons, ud):
        raise NotImplementedError("Cannot draw a Deconvolution with photon shooting")

    def _drawKImage(self, image):
        self.orig_obj._drawKImage(image)
        do_inverse = image.array > self._min_acc_kvalue
        image.array[do_inverse] = 1./image.array[do_inverse]
        image.array[~do_inverse] = self._inv_min_acc_kvalue

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_sbp',None)
        return d

    def __setstate__(self, d):
        self.__dict__ = d


def AutoConvolve(obj, real_space=None, gsparams=None):
    """A function for autoconvolving either a GSObject or ChromaticObject.

    This function will inspect its input argument to decide if a AutoConvolution object or a
    ChromaticAutoConvolution object is required to represent the convolution of a surface
    brightness profile with itself.

    @param obj              The object to be convolved with itself.
    @param real_space       Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the object has hard
                            edges.]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    @returns a AutoConvolution or ChromaticAutoConvolution instance as appropriate.
    """
    from .chromatic import ChromaticAutoConvolution
    if isinstance(obj, ChromaticObject):
        return ChromaticAutoConvolution(obj, real_space=real_space, gsparams=gsparams)
    elif isinstance(obj, GSObject):
        return AutoConvolution(obj, real_space=real_space, gsparams=gsparams)
    else:
        raise TypeError("Argument to AutoConvolve must be either a GSObject or a ChromaticObject.")


class AutoConvolution(Convolution):
    """A special class for convolving a GSObject with itself.

    It is equivalent in functionality to `Convolve([obj,obj])`, but takes advantage of
    the fact that the two profiles are the same for some efficiency gains.

    Initialization
    --------------

    The normal way to use this class is to use the AutoConvolve() factory function:

        >>> psf_sq = galsim.AutoConvolve(psf)

    @param obj              The object to be convolved with itself.
    @param real_space       Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the object has hard
                            edges.]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods
    -------

    There are no additional methods for AutoConvolution beyond the usual GSObject methods.
    """
    def __init__(self, obj, real_space=None, gsparams=None):
        if not isinstance(obj, GSObject):
            raise TypeError("Argument to AutoConvolution must be a GSObject.")

        # Check whether to perform real space convolution...
        # Start by checking if obj has a hard edge.
        hard_edge = obj.has_hard_edges

        if real_space is None:
            # The automatic determination is to use real_space if obj has hard edges.
            real_space = hard_edge
        elif bool(real_space) != real_space:
            raise TypeError("real_space must be a boolean")

        # Warn if doing DFT convolution for objects with hard edges.
        if not real_space and hard_edge:
            import warnings
            msg = """
            Doing auto-convolution of object with hard edges.
            This might be more accurate and/or faster using real_space=True"""
            warnings.warn(msg)

        # Can't do real space if object is not analytic, so check for that.
        if real_space and not obj.is_analytic_x:
            import warnings
            msg = """
            Object to be auto-convolved is not analytic in real space.
            Cannot use real space convolution.
            Switching to DFT method."""
            warnings.warn(msg)
            real_space = False

        # Save the construction parameters (as they are at this point) as attributes so they
        # can be inspected later if necessary.
        self._real_space = bool(real_space)
        self._orig_obj = obj
        self._gsparams = GSParams.check(gsparams, self._orig_obj.gsparams)

        # So we can use Convolve methods when there is no advantage to overloading.
        self._obj_list = [obj, obj]

    @lazy_property
    def _sbp(self):
        return _galsim.SBAutoConvolve(self.orig_obj._sbp, self._real_space, self.gsparams._gsp)

    @property
    def orig_obj(self): return self._orig_obj
    @property
    def real_space(self): return self._real_space

    @property
    def _noise(self):
        if self.orig_obj.noise is not None:
            import warnings
            warnings.warn("Unable to propagate noise in galsim.AutoConvolution")
        return None

    def __eq__(self, other):
        return (isinstance(other, AutoConvolution) and
                self.orig_obj == other.orig_obj and
                self.real_space == other.real_space and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.AutoConvolution", self.orig_obj, self.real_space, self.gsparams))

    def __repr__(self):
        return 'galsim.AutoConvolution(%r, real_space=%r, gsparams=%r)'%(
                self.orig_obj, self.real_space, self.gsparams)

    def __str__(self):
        s = 'galsim.AutoConvolve(%s'%self.orig_obj
        if self.real_space:
            s += ', real_space=True'
        s += ')'
        return s

    def _prepareDraw(self):
        self.orig_obj._prepareDraw()

    def _shoot(self, photons, ud):
        from .photon_array import PhotonArray
        self.orig_obj._shoot(photons, ud)
        photons2 = PhotonArray(len(photons))
        self.orig_obj._shoot(photons2, ud)
        photons.convolve(photons2, ud)


def AutoCorrelate(obj, real_space=None, gsparams=None):
    """A function for autocorrelating either a GSObject or ChromaticObject.

    This function will inspect its input argument to decide if a AutoCorrelation object or a
    ChromaticAutoCorrelation object is required to represent the correlation of a surface
    brightness profile with itself.

    @param obj              The object to be convolved with itself.
    @param real_space       Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the object has hard
                            edges.]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    @returns an AutoCorrelation or ChromaticAutoCorrelation instance as appropriate.
    """
    from .chromatic import ChromaticAutoCorrelation
    if isinstance(obj, ChromaticObject):
        return ChromaticAutoCorrelation(obj, real_space=real_space, gsparams=gsparams)
    elif isinstance(obj, GSObject):
        return AutoCorrelation(obj, real_space=real_space, gsparams=gsparams)
    else:
        raise TypeError("Argument to AutoCorrelate must be either a GSObject or a ChromaticObject.")


class AutoCorrelation(Convolution):
    """A special class for correlating a GSObject with itself.

    It is equivalent in functionality to
        galsim.Convolve([obj,obj.createRotated(180.*galsim.degrees)])
    but takes advantage of the fact that the two profiles are the same for some efficiency gains.

    This class is primarily targeted for use by the CorrelatedNoise models when convolving
    with a GSObject.

    Initialization
    --------------

    The normal way to use this class is to use the AutoCorrelate() factory function:

        >>> psf_sq = galsim.AutoCorrelate(psf)

    @param obj              The object to be convolved with itself.
    @param real_space       Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the object has hard
                            edges.]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]

    Methods
    -------

    There are no additional methods for AutoCorrelation beyond the usual GSObject methods.
    """
    def __init__(self, obj, real_space=None, gsparams=None):
        if not isinstance(obj, GSObject):
            raise TypeError("Argument to AutoCorrelation must be a GSObject.")

        # Check whether to perform real space convolution...
        # Start by checking if obj has a hard edge.
        hard_edge = obj.has_hard_edges

        if real_space is None:
            # The automatic determination is to use real_space if obj has hard edges.
            real_space = hard_edge
        elif bool(real_space) != real_space:
            raise TypeError("real_space must be a boolean")

        # Warn if doing DFT convolution for objects with hard edges.
        if not real_space and hard_edge:
            import warnings
            msg = """
            Doing auto-correlation of object with hard edges.
            This might be more accurate and/or faster using real_space=True"""
            warnings.warn(msg)

        # Can't do real space if object is not analytic, so check for that.
        if real_space and not obj.is_analytic_x:
            import warnings
            msg = """
            Object to be auto-correlated is not analytic in real space.
            Cannot use real space convolution.
            Switching to DFT method."""
            warnings.warn(msg)
            real_space = False

        # Save the construction parameters (as they are at this point) as attributes so they
        # can be inspected later if necessary.
        self._real_space = bool(real_space)
        self._orig_obj = obj
        self._gsparams = GSParams.check(gsparams, self._orig_obj.gsparams)

        # So we can use Convolve methods when there is no advantage to overloading.
        self._obj_list = [obj, obj.transform(-1,0,0,-1)]

    @lazy_property
    def _sbp(self):
        return _galsim.SBAutoCorrelate(self.orig_obj._sbp, self._real_space, self.gsparams._gsp)

    @property
    def orig_obj(self): return self._orig_obj
    @property
    def real_space(self): return self._real_space

    @property
    def _noise(self):
        if self.orig_obj.noise is not None:
            import warnings
            warnings.warn("Unable to propagate noise in galsim.AutoCorrelation")
        return None

    def __eq__(self, other):
        return (isinstance(other, AutoCorrelation) and
                self.orig_obj == other.orig_obj and
                self.real_space == other.real_space and
                self.gsparams == other.gsparams)

    def __hash__(self):
        return hash(("galsim.AutoCorrelation", self.orig_obj, self.real_space, self.gsparams))

    def __repr__(self):
        return 'galsim.AutoCorrelation(%r, real_space=%r, gsparams=%r)'%(
                self.orig_obj, self.real_space, self.gsparams)

    def __str__(self):
        s = 'galsim.AutoCorrelate(%s'%self.orig_obj
        if self.real_space:
            s += ', real_space=True'
        s += ')'
        return s

    def _prepareDraw(self):
        self._orig_obj._prepareDraw()

    def _shoot(self, photons, ud):
        from .photon_array import PhotonArray
        self.orig_obj._shoot(photons, ud)
        photons2 = PhotonArray(len(photons))
        self.orig_obj._shoot(photons2, ud)

        # Flip sign of (x, y) in one of the results
        photons2.scaleXY(-1)

        photons.convolve(photons2, ud)
