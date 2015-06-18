from astropy import coordinates as coord
from astropy import units as u


class Axes(object):
    """
    axes_type : str
        Type of axes. One of ['Celestial', 'Pixel', 'Spectral', 'Time']
    reference_frame : `~astropy.coordinates.BaseCoordinateFrame`
        Sky reference frame associated with this WCS
    order : tuple
        order of axes
    names : tuple of str
        names of axes
    unit : `~astropy.units.Unit`
    """
    def __init__(self, axes_type, reference_frame, order, names, unit):
        self._axes_type = axes_type
        self._reference_frame = reference_frame
        self._order = order
        self._names = names
        self._unit = unit

    @property
    def axes_type(self):
        return self._axes_type

    @property
    def reference_frame(self):
        return self._reference_frame

    @property
    def order(self):
        return self._order

    @property
    def names(self):
        return self._names

    @property
    def unit(self):
        return self._unit


class CelestialAxes(Axes):
    def __init__(self, reference_frame='ICRS', order=(0, 1), names=('RA', 'DEC'),
                unit=(u.deg, u.deg)):
        try:
            # try to create a reference frame object using astropy.coordinates
            reference_frame = getattr(coord.builtin_frames, reference_frame)
        except AttributeError:
            pass
        super(CelestialAxes, self).__init__("celestial", reference_frame, order=order,
                names=names, unit=unit) 

    def world_coordinates(self, x, y):
        return coord.SkyCoord(x*self.unit[0], y*self.unit[1], frame = self.reference_frame)


class SpectralAxes(Axes):
    def __init__(self, reference_frame=None,
                order=(0,), unit=None):
        super(SpectralAxes, self).__init__("spectral", reference_frame, order=order, unit=unit)


class PixelAxes(Axes):
    def __init__(self, reference_frame="Local", order=(0, 1), names=('x', 'y'),
        unit=(u.pixel, u.pixel)):
        super(PixelAxes, self).__init__("Pixel", reference_frame, order, names, unit)


class TimeAxes(Axes):
    def __init__(self, reference_frame, order=(0,), names=('time',), unit=u.s):
        super(TimeAxes, self).__init__("Time", reference_frame, order, names, unit)

