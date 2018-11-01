# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np

from astropy.io import fits
from astropy import wcs as fitswcs
from astropy.wcs import utils as awutils
from astropy.modeling.models import custom_model
from astropy.modeling.core import Model

from . import coordinate_frames as cf
from .wcs import WCS


class FITSDetector2World(Model):

    inputs = ('x', 'y')
    outputs = ('lon', 'lat')

    def __init__(self, filename, ext=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(filename, str):
            fobj = fits.open(filename)
            header = fobj[ext].header
        elif isinstance(filename, fits.HDUList):
            fobj = filename
            header = fobj[ext].header
        elif isinstance(filename, fits.Header):
            header = filename
            fobj = None
            
        else:
            raise TypeError("Expected input to be a filename path or an HDUList object")
        self.fitswcsobj= fitswcs.WCS(header, fobj)

        if fobj is not None:
            fobj.close()

    def evaluate(self, x, y):
        return self.fitswcsobj.all_pix2world(x, y, 0)

    @property
    def inverse(self):
        w2fits = self.fitswcsobj.to_fits(relax=True)
        return FITSWorld2Detector(w2fits, 0)


class FITSWorld2Detector(Model):

    inputs = ('x', 'y')
    outputs = ('lon', 'lat')

    def __init__(self, filename, ext, **kwargs):
        super().__init__(**kwargs)
        if isinstance(filename, str):
            fobj = fits.open(filename)
        elif isinstance(filename, fits.HDUList):
            fobj = filename
        else:
            raise TypeError("Expected input to be a filename path or an HDUList object")
        #fobj = fits.open(filename)
        self.fitswcsobj= fitswcs.WCS(fobj[ext].header, fobj)
        fobj.close()

    def evaluate(self, x, y):
        return self.fitswcsobj.all_world2pix(x, y, 0)

    @property
    def inverse(self):
        w2fits = self.fitswcsobj.to_fits(relax=True)
        return FITSDetector2World(w2fits, 0)

    
def wcs_from_fits(filename, ext, **kwargs):
    #fobj = fits.open(filename)
    #fwcs = fitswcs.WCS(fobj[ext].header, fobj, **kwargs)
    fwcs = FITSDetector2World(filename, ext, name="fitswcs")
    output_frame = out_frame_from_fits(fwcs.fitswcsobj)
    input_frame = cf.Frame2D(name="detector")

    '''
    #if fwcs.has_distortion():
    if not np.isclose(fwcs.fitswcsobj.pix2foc(1,1,1), (1,1)).all():
        detector2focal = fwcs.pix2foc
        focal2world = fwcs.wcs_pix2world
        focal = cf.Frame2D(name='focal', unit=("",""))
        pipeline = [(input_frame, detector2focal),
                    (focal, focal2world),
                    (output_frame, None)]
    else:
    '''
    #detector2world = fwcs.all_pix2world
    pipeline = [(input_frame, fwcs),
                (output_frame, None)]

    return WCS(pipeline)


def out_frame_from_fits(fwcs):
    """Construct an output frame from a FITSWCS object."""
    frame = awutils.wcs_to_celestial_frame(fwcs)
    output_frame = cf.CelestialFrame(name="world", reference_frame=frame)
    return output_frame
