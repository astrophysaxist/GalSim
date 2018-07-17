# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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

import numpy as np

from . import _galsim
from .gsparams import GSParams
from .gsobject import GSObject
from .position import PositionD
from .utilities import lazy_property, doc_inherit
from .errors import (
    GalSimRangeError,
    GalSimValueError,
    GalSimIncompatibleValuesError,
    convert_cpp_errors,
)
from .random import UniformDeviate
from .gaussian import Gaussian

class RandomWalk(GSObject):
    """
    A class for generating a set of point sources distributed using a random
    walk of points drawn from a specified distribution.

    Uses of this profile include representing an "irregular" galaxy, or
    adding this profile to an Exponential to represent knots of star formation.

    Random walk profiles have "shape noise" that depends on the number of point
    sources used.  For example, using the default Gaussian distribution, with
    100 points, the shape noise is g~0.05, and this will decrease as more
    points are added.  The profile can be sheared to give additional
    ellipticity, for example to follow that of an associated disk.

    We use the analytic approximation of an infinite number of steps, which is
    a good approximation even if the desired number of steps were less than 10.

    The requested half light radius (hlr) should be thought of as a rough
    value.  With a finite number point sources the actual realized hlr will be
    noisy.

    Initialization
    --------------
    @param  npoints                 Number of point sources to generate.
    @param  half_light_radius       Optional half light radius of the
                                    distribution of points.  This value is used
                                    for a Gaussian distribution if an explicit
                                    profile is not sent. This is the mean half
                                    light radius produced by an infinite number
                                    of points.  A single instance will be noisy.
                                    [default None]
    @param  flux                    Optional total flux in all point sources.
                                    This value is used for a Gaussian distribution
                                    if an explicit profile is not sent. Defaults
                                    to None if profile is sent, otherwise 1.
                                    [default: 1]
    @param  profile                 Optional profile to use for drawing points.
                                    If a profile is sent, the half_light_radius
                                    and flux keywords are ignored.
                                    [default: None]
    @param  points                  The points to use.  In this case, the
                                    input half_light_radius and the profile
                                    is not used to generate points, but
                                    one of those must still be sent to create
                                    a consistent RandomWalk object.
    @param  rng                     Optional random number generator. Can be
                                    any galsim.BaseDeviate.  If None, the rng
                                    is created internally.
                                    [default: None]
    @param  gsparams                Optional GSParams for the objects
                                    representing each point source.
                                    [default: None]

    Methods
    -------

    This class inherits from galsim.Sum. Additional methods are

        calculateHLR:
            Calculate the actual half light radius of the generated points
        set_points:
            Set the points used. This over-rides any existing points.

    There are also "getters",  implemented as read-only properties

        .npoints
        .input_half_light_radius
        .flux
        .points
            The array of x,y offsets used to create the point sources

    And setters
        .points: calls set_points

    Notes
    -----

    - The algorithm is a modified version of that presented in

          https://arxiv.org/abs/1312.5514v3

      Modifications are
        1) there is no outer cutoff to how far a point can wander
        2) We use the approximation of an infinite number of steps.
    """
    # these allow use in a galsim configuration context

    #_req_params = { "npoints" : int }
    _opt_params = {
        "npoints" : int,
        "flux" : float ,
        "half_light_radius": float,
        "profile": GSObject,
    }
    _single_params = []
    _takes_rng = True

    _has_hard_edges = False
    _is_axisymmetric = False
    _is_analytic_x = False
    _is_analytic_k = True

    def __init__(self, npoints=None, **kw):
        from .random import BaseDeviate

        rng=kw.pop('rng',None)
        self._set_rng(rng)

        gsparams=kw.pop('gsparams',None)
        self._gsparams = GSParams.check(gsparams)

        profile=kw.pop('profile',None)

        # the user can bypass all other mechanisms and just
        # set the points directly
        input_points=kw.pop('points',None)

        if profile is None:

            if ('half_light_radius' not in kw
                    or kw['half_light_radius'] is None):
                raise GalSimIncompatibleValuesError(
                    "send a half_light_radius "
                    "when not sending a profile"
                )

            if 'flux' not in kw or kw['flux'] is None:
                self._flux = 1.0
            else:
                self._flux=float(kw['flux'])
                if self._flux < 0.0:
                    raise GalSimRangeError("flux must be >= 0", self._flux, 0.)

            half_light_radius=float(kw['half_light_radius'])

            profile=Gaussian(
                half_light_radius=half_light_radius,
                flux=self._flux,
            )

            self._half_light_radius=half_light_radius

            self._set_gaussian_rng()
        else:
            if 'flux' in kw and kw['flux'] is not None:
                raise GalSimIncompatibleValuesError(
                    "don't send a flux "
                    "when sending a profile"
                )
            if ('half_light_radius' in kw
                    and kw['half_light_radius'] is not None):
                raise GalSimIncompatibleValuesError(
                    "don't send a half_light_radius "
                    "when sending a profile"
                )

            if not isinstance(profile, GSObject):
                raise GalSimIncompatibleValuesError("profile must be a GSObject")

            # the half light radius is not used
            try:
                # not all GSObjects have this attribute
                self._half_light_radius = profile.half_light_radius
            except:
                self._half_light_radius=None

            self._flux=profile.flux

        self._profile=profile

        self.set_points(points=input_points, npoints=npoints)

    @property
    def points(self):
        """
        get a copy of the points
        """
        return self._get_points().copy()

    def _get_points(self):
        """
        get a reference to the points
        """
        if not hasattr(self,'_points'):
            self._points=self._generate_points(self._npoints)

        return self._points


    @points.setter
    def points(self, points):
        """
        set the points. See the set_points() method for documentation
        """
        return self.set_points(points=points)

    @lazy_property
    def _sbp(self):
        fluxper=self._flux/self._npoints
        deltas = []

        points=self._get_points()
        with convert_cpp_errors():
            for p in points:
                d = _galsim.SBDeltaFunction(fluxper, self.gsparams._gsp)
                d = _galsim.SBTransform(d, 1.0, 0.0, 0.0, 1.0, _galsim.PositionD(p[0],p[1]), 1.0,
                                        self.gsparams._gsp)
                deltas.append(d)
            return _galsim.SBAdd(deltas, self.gsparams._gsp)

    @property
    def input_half_light_radius(self):
        """
        Get the input half light radius (HLR).

        Note the input HLR is not necessarily the realized HLR,
        due to the finite number of points used in the profile.

        If a profile is sent, and that profile is a Transformation object (e.g.
        it has been sheared, its flux set, etc), then this value will be None.

        You can get the *calculated* half light radius using the calculateHLR
        method.  That value will be valid in all cases.
        """
        return self._half_light_radius

    @property
    def npoints(self):
        """
        get the number of points
        """
        return self._npoints

    def calculateHLR(self):
        """
        calculate the half-light radius of the generated points
        """
        points = self._get_points()
        my,mx=points.mean(axis=0)

        r=np.sqrt( (points[:,0]-my)**2 + (points[:,1]-mx)**2)

        hlr=np.median(r)

        return hlr

    def _set_gaussian_rng(self):
        """
        Set the random number generator used to create the points

        We are approximating the random walk to have infinite number
        of steps, which is just a gaussian
        """
        from .random import GaussianDeviate
        # gaussian step size in each dimension for a random walk with infinite
        # number steps
        self._sigma_step = self._half_light_radius/2.3548200450309493*2
        self._gauss_rng = GaussianDeviate(self._rng, sigma=self._sigma_step)

    def set_points(self, points=None, npoints=None):
        """
        set the points, either using the input points or
        generating the number of specified points npoints

        Note when npoints= is sent, the generation of points is lazy.   This is
        important when transformations are to be applied to the profile later,
        which will always clear any existing set of points.

        points
        ------
        array: optional
            (npoints, 2) array.
        npoints: number optional
            number of new points to generate. Existing points
            will be replaced
        """

        if npoints is None and points is None:
            raise GalSimIncompatibleValuesError(
                "send either points= or npoints="
            )

        if points is not None:
            points = np.array(points, dtype='f8', copy=False)

            if len(points.shape) != 2:
                raise GalSimIncompatibleValuesError(
                    "input points should be an array-like with shape (npoints,2)"
                    "got shape %s" % str(points.shape)
                )


            self._points=points
            self._npoints=self._points.shape[0]
        else:
            try:
                npoints = int(npoints)
            except ValueError as err:
                raise GalSimValueError("npoints should be a number: %s", str(err))

            if npoints <= 0:
                raise GalSimRangeError("npoints must be > 0", npoints, 1)

            self._npoints=npoints


    def _generate_points(self, npoints):
        """
        We must use a galsim random number generator, in order for
        this profile to be used in the configuration file context.
        """


        ud = UniformDeviate(self._rng)
        photons = self._profile.shoot(npoints, ud)
        points = np.column_stack([ photons.x, photons.y ])

        return points

    def _set_rng(self, rng):
        """
        type and range checking on the inputs
        """
        from .random import BaseDeviate

        self._rng=rng

        if self._rng is None:
            self._rng = BaseDeviate()

        if not isinstance(self._rng, BaseDeviate):
            raise TypeError("rng must be an instance of "
                            "galsim.BaseDeviate, got %s"%self._rng)

    def __str__(self):
        rep='galsim.RandomWalk(%(npoints)d, profile=%(profile)s, gsparams=%(gsparams)s,rng=%(rng)s, points=%(points)s)'

        points=self._get_points()
        rep = rep % dict(
            npoints=self._npoints,
            profile=repr(self._profile),
            gsparams=repr(self.gsparams),
            rng=repr(self._rng),
            points=str(points[0:1]).replace(']]','],...]'),
        )

        return rep

    def __repr__(self):
        rep='galsim.RandomWalk(%(npoints)d, profile=%(profile)s, gsparams=%(gsparams)s,rng=%(rng)s, points=%(points)s)'

        points=self._get_points()
        prepr=repr(points).replace('array(','').replace(')','')
        rep = rep % dict(
            npoints=self._npoints,
            profile=repr(self._profile),
            gsparams=repr(self.gsparams),
            rng=repr(self._rng),
            points=prepr,
        )

        return rep

    def __eq__(self, other):
        return (
            isinstance(other, RandomWalk) and
            self._npoints == other._npoints and
            self._half_light_radius == other._half_light_radius and
            self._flux == other._flux and
            self.gsparams == other.gsparams and
            np.allclose(self._get_points(), other._get_points())
        )

    def __hash__(self):
        vals=(
            "galsim.RandomWalk",
            self._npoints,
            self._half_light_radius,
            self._flux,
            self.gsparams,
        )
        return hash(vals)

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_sbp',None)
        return d

    def __setstate__(self, d):
        self.__dict__ = d

    @property
    def _maxk(self):
        return self._sbp.maxK()

    @property
    def _stepk(self):
        return self._sbp.stepK()

    @property
    def _centroid(self):
        return PositionD(self._sbp.centroid())

    @property
    def _positive_flux(self):
        return self._sbp.getPositiveFlux()

    @property
    def _negative_flux(self):
        return self._sbp.getNegativeFlux()

    @property
    def _max_sb(self):
        return self._sbp.maxSB()

    @doc_inherit
    def _kValue(self, kpos):
        return self._sbp.kValue(kpos._p)

    @doc_inherit
    def _shoot(self, photons, ud):
        self._sbp.shoot(photons._pa, ud._rng)

    @doc_inherit
    def _drawKImage(self, image):
        self._sbp.drawK(image._image, image.scale)


    """
    these methods over-ride the usual GSObject methods.

    Because point sources are being used, it is more efficient to generate a
    new set of points drawn from the transformed profile.
    """

    def shear(self, *args, **kwargs):
        """
        Special method override.  Because point sources are being used, it is
        more efficient to generate a new set of points drawn from the
        transformed profile.
        """
        return self.__class__(
            npoints=self.npoints,
            profile=self._profile.shear(*args, **kwargs),
        )

    def shift(self, *args, **kwargs):
        """
        Special method override.  Because point sources are being used, it is
        more efficient to generate a new set of points drawn from the
        transformed profile.
        """
        return self.__class__(
            npoints=self.npoints,
            profile=self._profile.shift(*args, **kwargs),
        )


    def dilate(self, *args):
        """
        Special method override.  Because point sources are being used, it is
        more efficient to generate a new set of points drawn from the
        transformed profile.
        """
        return self.__class__(
            npoints=self.npoints,
            profile=self._profile.dilate(*args)
        )

    def magnify(self, *args):
        """
        Special method override.  Because point sources are being used, it is
        more efficient to generate a new set of points drawn from the
        transformed profile.
        """
        return self.__class__(
            npoints=self.npoints,
            profile=self._profile.magnify(*args)
        )


    def expand(self, *args):
        """
        Special method override.  Because point sources are being used, it is
        more efficient to generate a new set of points drawn from the
        transformed profile.
        """
        return self.__class__(
            npoints=self.npoints,
            profile=self._profile.expand(*args)
        )

    def lens(self, *args):
        """
        Special method override.  Because point sources are being used, it is
        more efficient to generate a new set of points drawn from the
        transformed profile.
        """
        return self.__class__(
            npoints=self.npoints,
            profile=self._profile.lens(*args)
        )

    def rotate(self, *args):
        """
        Special method override.  Because point sources are being used, it is
        more efficient to generate a new set of points drawn from the
        transformed profile.
        """
        return self.__class__(
            npoints=self.npoints,
            profile=self._profile.rotate(*args)
        )

    def transform(self, *args):
        """
        Special method override.  Because point sources are being used, it is
        more efficient to generate a new set of points drawn from the
        transformed profile.
        """
        return self.__class__(
            npoints=self.npoints,
            profile=self._profile.transform(*args)
        )

    def withFlux(self, *args):
        """
        Special method override. Use the same set of points, but modify the
        flux
        """
        return self.__class__(
            points=self.points,
            profile=self._profile.withFlux(*args)
        )

    def withScaledFlux(self, *args):
        """
        Special method override. Use the same set of points, but modify the
        flux
        """
        return self.__class__(
            points=self.points,
            profile=self._profile.withScaledFlux(*args)
        )


    '''
    def withGSParams(self, *args):
        """
        Special method override for RandomWalk objects.  Because point sources
        are being used, it is more efficient to generate a new set of points
        drawn from the transformed profile.
        """
        return RandomWalk(
            npoints=self.npoints,
            profile=self._profile.withGSParams(*args)
        )
    '''

