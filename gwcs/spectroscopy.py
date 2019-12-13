# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Spectroscopy related models.
"""

import numpy as np
from astropy.modeling.core import Model
from astropy.modeling.parameters import Parameter
import astropy.units as u


__all__ = ['ToDirectionCosines', 'FromDirectionCosines',
           'WavelengthFromGratingEquation', 'AnglesFromGratingEquation3D',
           'Snell3D', 'SellmeierGlass']


class ToDirectionCosines(Model):
    """
    Transform a vector to direction cosines.
    """
    _separable = False

    n_inputs = 3
    n_outputs = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs = ('x', 'y', 'z')
        self.outputs = ('cosa', 'cosb', 'cosc', 'length')

    def evaluate(self, x, y, z):
        vabs = np.sqrt(1. + x**2 + y**2)
        cosa = x / vabs
        cosb = y / vabs
        cosc = 1. / vabs
        return cosa, cosb, cosc, vabs

    def inverse(self):
        return FromDirectionCosines()


class FromDirectionCosines(Model):
    """
    Transform directional cosines to vector.
    """
    _separable = False

    n_inputs = 4
    n_outputs = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs = ('cosa', 'cosb', 'cosc', 'length')
        self.outputs = ('x', 'y', 'z')

    def evaluate(self, cosa, cosb, cosc, length):

        return cosa * length, cosb * length, cosc * length

    def inverse(self):
        return ToDirectionCosines()


class WavelengthFromGratingEquation(Model):
    r""" Solve the Grating Dispersion Law for the wavelength.

    .. Note:: This form of the equation can be used for paraxial
    (small angle approximation) as well as oblique incident angles.
    With paraxial systems the inputs are sin of the angles and it
    transforms to :math:`\sin(alpha_in) + \sin(alpha_out) / m * d` .
    With oblique angles the inputs are the direction cosines of the
    angles.

    Parameters
    ----------
    groove_density : int
        Grating ruling density in units of 1/length.
    spectral_order : int
        Spectral order.

    Examples
    --------
    >>> from astropy.modeling.models import math
    >>> model = WavelengthFromGratingEquation(groove_density=20000*1/u.m, spectral_order=-1)
    >>> alpha_in = (math.Deg2radUfunc() | math.SinUfunc())(.0001 * u.deg)
    >>> alpha_out = (math.Deg2radUfunc() | math.SinUfunc())(.0001 * u.deg)
    >>> lam = model(alpha_in, alpha_out)
    >>> print(lam)
    -1.7453292519934437e-10 m

    """

    _separable = False

    linear = False

    n_inputs = 2
    n_outputs = 1

    groove_density = Parameter(default=1)
    """ Grating ruling density in units of 1/m."""
    spectral_order = Parameter(default=1)
    """ Spectral order."""

    def __init__(self, groove_density, spectral_order, **kwargs):
        super().__init__(groove_density=groove_density,
                         spectral_order=spectral_order, **kwargs)
        self.inputs = ("alpha_in", "alpha_out")
        """ Sine function of the angles or the direction cosines."""
        self.outputs = ("wavelength",)
        """ Wavelength."""

    def evaluate(self, alpha_in, alpha_out, groove_density, spectral_order):
        return (alpha_in + alpha_out) / (groove_density * spectral_order)

    @property
    def return_units(self):
        if self.groove_density.unit is None:
            return None
        else:
            return {'wavelength': u.Unit(1 / self.groove_density.unit)}


class AnglesFromGratingEquation3D(Model):
    """
    Solve the 3D Grating Dispersion Law in Direction Cosine
    space for the refracted angle.

    Parameters
    ----------
    groove_density : int
        Grating ruling density in units of 1/m.
    order : int
        Spectral order.

    Examples
    --------
    >>> from astropy.modeling.models import math
    >>> model = AnglesFromGratingEquation3D(groove_density=20000*1/u.m, spectral_order=-1)
    >>> alpha_in = (math.Deg2radUfunc() | math.SinUfunc())(.0001 * u.deg)
    >>> beta_in = (math.Deg2radUfunc() | math.SinUfunc())(.0001 * u.deg)
    >>> lam = 2e-6 * u.m
    >>> alpha_out, beta_out, gamma_out = model(lam, alpha_in, beta_in)
    >>> print(alpha_out, beta_out, gamma_out)
    0.04000174532925199 -1.7453292519934436e-06 0.9991996098716049

    """

    _separable = False

    linear = False

    n_inputs = 3
    n_outputs = 3

    groove_density = Parameter(default=1)
    """ Grating ruling density in units 1/ length."""

    spectral_order = Parameter(default=1)
    """ Spectral order."""

    def __init__(self, groove_density, spectral_order, **kwargs):
        super().__init__(groove_density=groove_density,
                         spectral_order=spectral_order, **kwargs)
        self.inputs = ("wavelength", "alpha_in", "beta_in")
        """ Wavelength and 2 angle coordinates going into the grating."""

        self.outputs = ("alpha_out", "beta_out", "gamma_out")
        """ Two angles coming out of the grating. """

    def evaluate(self, wavelength, alpha_in, beta_in,
                 groove_density, spectral_order):
        if alpha_in.shape != beta_in.shape:
            raise ValueError("Expected input arrays to have the same shape.")

        if isinstance(groove_density, u.Quantity):
            alpha_in = u.Quantity(alpha_in)
            beta_in = u.Quantity(beta_in)

        alpha_out = -groove_density * spectral_order * wavelength + alpha_in
        beta_out = - beta_in
        gamma_out = np.sqrt(1 - alpha_out ** 2 - beta_out ** 2)
        return alpha_out, beta_out, gamma_out

    @property
    def input_units(self):
        if self.groove_density.unit is None:
            return None
        else:
            return {'wavelength': 1 / self.groove_density.unit,
                    'alpha_in': u.Unit(1),
                    'beta_in': u.Unit(1)}


class Snell3D(Model):
    _separable = False

    linear = False

    n_inputs = 3
    n_outputs = 3

    n = Parameter(default=1)

    def __init__(self, n=n, **kwargs):
        super().__init__(n=n, **kwargs)
        self.inputs = ('wavelength', 'alpha_in', 'beta_in', 'gamma_in')
        self.outputs = ('alpha_out', 'beta_out', 'gamma_out')

    @staticmethod
    def evaluate(alpha_in, beta_in, gamma_in, n):
        # Apply Snell's law through front surface, eq 5.3.3 II
        xout = alpha_in / n
        yout = beta_in / n
        zout = np.sqrt(1.0 - xout**2 - yout**2)
        return xout, yout, zout

    @property
    def input_units(self):
        return {'alpha_in': u.rad,
                'beta_in': u.rad,
                'gamma_in': u.rad}


class SellmeierGlass(Model):
    """
    Sellmeier equation for glass.

     Parameters
     ----------
     B_coef : ndarray
         Iterable of size 3 containing B coefficients.
     C_coef : ndarray
         Iterable of size 3 containing c coefficients in
         units of u.um**2 .

     Examples
     --------
     >>> import astropy.units as u
     >>> b_coef = [0.58339748, 0.46085267, 3.8915394]
     >>> c_coef = [0.00252643, 0.010078333, 1200.556] * u.um**2
     >>> model = SellmeierGlass(b_coef, c_coef)
     >>> model(2 * u.m)
         <Quantity 2.43634758>

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Sellmeier_equation

    Notes
    -----
    Model formula:

        .. math::

            n(\lambda)^2 = 1 + \frac{(B1 * \lambda^2 )}{(\lambda^2 - C1)} +
            \frac{(B2 * \lambda^2 )}{(\lambda^2 - C2)} +
            \frac{(B3 * \lambda^2 )}{(\lambda^2 - C3)}

    """
    _separable = False
    standard_broadcasting = False
    linear = False

    n_inputs = 1
    n_outputs = 1

    B_coef = Parameter(default=np.array([1, 1, 1]))
    """ B1, B2, B3 coefficients. """
    C_coef = Parameter(default=np.array([0, 0, 0]))
    """ C1, C2, C3 coefficients in units of um ** 2. """


    def __init__(self, B_coef, C_coef, **kwargs):
        super().__init__(B_coef, C_coef)
        self.inputs = ('wavelength',)
        self.outputs = ('n',)

    @staticmethod
    def evaluate(wavelength, B_coef, C_coef):
        B1, B2, B3 = B_coef[0]
        C1, C2, C3 = C_coef[0]

        n = np.sqrt(1. +
                    B1 * wavelength ** 2 / (wavelength ** 2 - C1) +
                    B2 * wavelength ** 2 / (wavelength ** 2 - C2) +
                    B3 * wavelength ** 2 / (wavelength ** 2 - C3)
                    )
        return n

    @property
    def input_units(self):
        if self.C_coef.unit is None:
            return None
        else:
            return {'wavelength': u.um}


class SellmeierZemax(Model):
    """
    Sellmeier equation used by Zemax.
    """
    _separable = False
    standard_broadcasting = False
    linear = False

    n_inputs = 1
    n_outputs = 1

    temp = Parameter(default=0)
    ref_temp = Parameter(default=0)
    ref_pressure = Parameter(default=0)
    pressure = Parameter(default=0)
    # B1 = Parameter(default=1)
    # B2 = Parameter(default=1)
    # B3 = Parameter(default=1)
    # C1 = Parameter(default=1)
    # C2 = Parameter(default=1)
    # C3 = Parameter(default=1)
    B_coef = Parameter(default=[1, 1, 1])
    C_coef = Parameter(default=[0, 0, 0])
    D_coef = Parameter(default=[0, 0, 0])
    E_coef = Parameter(default=[1, 1, 1])

    def __init__(self, temp=temp, ref_temp=ref_temp, ref_pressure=ref_pressure,
                 pressure=pressure, B_coef=B_coef, C_coef=C_coef, D_coef=D_coef,
                 E_coef=E_coef, **kwargs):

        # self.temp = temp
        # self.ref_temp = ref_temp
        # self.ref_pressure = ref_pressure
        # self.pressure = pressure
        # self.kcoef = kcoef
        # self.lcoef = lcoef
        # self.tcoef = tcoef
        super().__init__(**kwargs)
        self.inputs = ('wavelength',)
        self.outputs =('n',)

    def evaluate(self, wavelength, temp, ref_temp, ref_presssure,
                 pressure, B_coef, C_coef, D_coef, E_coef):
        # Convert to microns
        #lam = np.asarray(wavelength * 1e6)
        # temp = self.temp
        # ref_temp = self.ref_temp
        #KtoC = 273.15  # kelvin to celcius conversion
        #temp -= KtoC
        #ref_temp -= KtoC
        if isinstance(temp, u.Quantity):
            temp = temp.to(u.Celsius)
            ref_temp = ref_temp.to(u.Celsius)
        delt = temp - ref_temp

        K1, K2, K3 = kcoef
        L1, L2, L3 = lcoef
        D0, D1, D2 = D_coef
        E0, E1, lam_tk = E_coef
        # pressure = self.pressure
        # ref_pressure = self.ref_pressure

        nref = 1. + (6432.8 + 2949810. * lam ** 2 /
                    (146.0 * lam ** 2 - 1.) + (5540.0 * lam ** 2) /
                    (41.0 * lam ** 2 - 1.)) * 1e-8

        # T should be in C, P should be in ATM
        nair_obs = 1.0 + ((nref - 1.0) * pressure) / (1.0 + (temp - 15.) * 3.4785e-3)
        nair_ref = 1.0 + ((nref - 1.0) * ref_pressure) / (1.0 + (ref_temp - 15) * 3.4785e-3)

        # Compute the relative index of the glass at Tref and Pref using Sellmeier equation I.
        lamrel = lam * nair_obs / nair_ref

        nrel = SellmeierGlass.evaluate(lam, [self.kcoef], [self.lcoef])

        # Convert the relative index of refraction at the reference temperature and pressure
        # to absolute.
        nabs_ref = nrel * nair_ref

        # Compute the absolute index of the glass
        delnabs = (0.5 * (nrel ** 2 - 1.) / nrel) * \
                (D0 * delt + D1 * delt ** 2 + D2 * delt ** 3 + \
                 (E0 * delt + E1 * delt ** 2) / (lamrel ** 2  - lam_tk ** 2))
        nabs_obs = nabs_ref + delnabs

        # Define the relative index at the system's operating T and P.
        n = nabs_obs / nair_obs
        return n
