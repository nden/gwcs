# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, unicode_literals, print_function

import copy
import functools
import numpy as np
from astropy.extern import six
from astropy.io import fits
from astropy.modeling import models
from astropy.modeling.core import Model
from astropy.utils import isiterable

from . import axes
from .util import ModelDimensionalityError, AxesError
from .selector import *


__all__ = ['WCS']


class WCS(object):

    """
    Basic WCS class.

    Parameters
    ----------
    output_axes : str, `~gwcs.axes.Axes`
        A coordinates object or a string name.
    input_axes : str, `~gwcs.axes.Axes`
        A coordinates object or a string name.
    forward_transform : `~astropy.modeling.Model` or a list
    A model to do the transform between ``input_axes`` and ``output_axes``.
        A list of (axes, transform) tuples where ``axes`` is the starting axes and
        ``transform`` is the transform from this axes to the next one or ``output_axes``.
    name : str
        a name for this WCS
    """

    def __init__(self, output_axes, input_axes='detector',
                 forward_transform=None, name=""):
        self._axes = {}
        self._pipeline = []
        self._input_axes, axes_obj = self._get_axes_name(input_axes
        self._coord_axes[self._input_axes] = axes_obj
        self._output_axes axes_obj = self._get_axes_name(output_axes)
        self._coord_axes[self._output_axes] = axes_obj
        self._name = name
        if forward_transform is not None:
            if isinstance(forward_transform, Model):
                self._pipeline = [(self._input_axes, forward_transform.copy()),
                                  (self._output_axes, None)]
            elif isinstance(forward_transform, list):
                for item in forward_transform:
                    name, axes_obj = self._get_axes_name(item[0])
                    self._coord_axes[name] = copy.deepcopy(axes_obj)
                    self._pipeline.append((name, item[1]))
            else:
                raise TypeError("Expected forward_transform to be a model or a "
                                "(axes, transform) list, got {0}".format(
                                    type(forward_transform)))
        else:
            self._pipeline = [(self._input_axes, None),
                              (self._output_axes, None)]

    def get_transform(self, from_axes, to_axes):
        """
        Return a transform between two coordinate axes.

        Parameters
        ----------
        from_axes : str or `~gwcs.axes.Axes`
            Initial coordinate axes.
        to_axes : str, or instance of `~gwcs.axes.Axes`
            Coordinate axes into which to transform.

        Returns
        -------
        transform : `~astropy.modeling.Model`
            Transform between two axes.
        """
        if not self._pipeline:
            return None
        from_name, from_obj = self._get_axes_name(from_axes)
        to_name, to_obj = self._get_axes_name(to_axes)

        # if from_name not in self.available_frames:
        #raise ValueError("Frame {0} is not in the available axes".format(from_axes))
        # if to_name not in self.available_frames:
        #raise ValueError("Frame {0} is not in the available axes".format(to_axes))
        try:
            from_ind = self._get_axes_index(from_name)
        except ValueError:
            raise AxesError("Axes {0} is not in the available axes".format(from_name))
        try:
            to_ind = self._get_axes_index(to_name)
        except ValueError:
            raise AxesError("Axes {0} is not in the available axes".format(to_name))
        transforms = np.array(self._pipeline[from_ind: to_ind])[:, 1].tolist()
        return functools.reduce(lambda x, y: x | y, transforms)

    def set_transform(self, from_axes, to_axes, transform):
        """
        Set/replace the transform between two coordinate axes.

        Parameters
        ----------
        from_axes : str or `~gwcs.axes.Axes`
            Initial coordinate axes.
        to_axes : str, or instance of `~gwcs.axes.Axes`
            Coordinate axes into which to transform.
        transform : `~astropy.modeling.Model`
            Transform between two axes.
        """
        from_name, from_obj = self._get_axes_name(from_axes)
        to_name, to_obj = self._get_axes_name(to_axes)
        if not self._pipeline:
            if from_name != self._input_axes:
                raise AxesError(
                    "Expected 'from_axes' to be {0}".format(self._input_axes))
            if to_axes != self._output_axes:
                raise AxesError(
                    "Expected 'to_axes' to be {0}".format(self._output_axes))
        try:
            from_ind = self._get_axes_index(from_name)
        except ValueError:
            raise AxesError("Axes {0} is not in the available axes".format(from_name))
        try:
            to_ind = self._get_axes_index(to_name)
        except ValueError:
            raise AxesError("Axes {0} is not in the available axes".format(to_name))

        if from_ind + 1 != to_ind:
            raise ValueError("Axes {0} and {1} are not  in sequence".format(from_name, to_name))
        self._pipeline[from_ind] = (self._pipeline[from_ind], transform)

    @property
    def forward_transform(self):
        """
        Return the total forward transform - from input to output coordinate axes.

        """

        if self._pipeline:
            if self._pipeline[-1] != (self._output_axes, None):
                self._pipeline.append((self._output_axes, None))
            return functools.reduce(lambda x, y: x | y, [step[1] for step in self._pipeline[: -1]])
        else:
            return None

    @property
    def backward_transform(self):
        """
        Return the total backward transform if available - from output to input coordinate system.

        Raises
        ------
        NotImplementedError :
            An analytical inverse does not exist.

        """
        backward = self.forward_transform.inverse
        return backward

    def _get_axes_index(self, axes):
        """
        Return the index in the pipeline where this axes is locate.
        """
        return np.asarray(self._pipeline)[:, 0].tolist().index(axes)

    def _get_axes_name(self, axes):
        """
        Return the name of the axes and a ``Axes`` object.

        Parameters
        ----------
        axes : str, `~gwcs.axes.Axes`
            Coordinate axes.

        Returns
        -------
        name : str
            The name of the axes.
        axes_obj : `~gwcs.axes.Axes`
            Axes instance or None (if `axes` is str)
        """
        if isinstance(axes, six.string_types):
            name = axes
            axes_obj = None
        else:
            name = axes.name
            axes_obj = axes
        return name, axes_obj

    def __call__(self, *args):
        """
        Executes the forward transform.

        args : float or array-like
            Inputs in the input coordinate system, separate inputs for each dimension.

        """
        if self.forward_transform is not None:
            return self.forward_transform(*args)

    def invert(self, *args, **kwargs):
        """
        Invert coordnates.

        The analytical inverse of the forward transform is used, if available.
        If not an iterative method is used.

        Parameters
        ----------
        args : float or array like
            coordinates to be inverted
        kwargs : dict
            keyword arguments to be passed to the iterative invert method.
        """
        try:
            return self.forward_transform.inverse(*args)
        except (NotImplementedError, KeyError):
            return self._invert(*args, **kwargs)

    def _invert(self, *args, **kwargs):
        """
        Implement iterative inverse here.
        """
        raise NotImplementedError

    def transform(self, from_axes, to_axes, *args):
        """
        Transform potitions between two coordinate axes.


        Parameters
        ----------
        from_axes : str or `~gwcs.axes.Axes`
            Initial coordinate axes.
        to_axes : str, or instance of `~gwcs.axes.Axes`
            Coordinate axes into which to transform.
        args : float
            input coordinates to transform
        """
        transform = self.get_transform(from_axes, to_axes)
        return transform(*args)

    @property
    def available_axes(self):
        """
        List all axes in this WCS object.

        Returns
        -------
        available_axes : dict
            {axes_name: axes_object or None}
        """
        return self._coord_axes

    def insert_transform(self, axes, transform, after=False):
        """
        Insert a transform before (default) or after a coordinate axes.

        Append (or prepend) a transform to the transform connected to axes.

        Parameters
        ----------
        axes : str or `~gwcs.axes.Axes`
            Coordinate axes which sets the point of insertion.
        transform : `~astropy.modeling.Model`
            New transform to be inserted in the pipeline
        after : bool
            If True, the new transform is inserted in the pipeline
            immediately after `axes`.
        """
        name, _ = self._get_axes_name(axes)
        axes_ind = self._get_axes_index(name)
        if not after:
            fr, current_transform = self._pipeline[axes_ind - 1]
            self._pipeline[axes_ind - 1] = (fr, current_transform | transform)
        else:
            fr, current_transform = self._pipeline[axes_ind]
            self._pipeline[axes_ind] = (fr, transform | current_transform)

    @property
    def unit(self):
        """The unit of the coordinates in the output coordinate system."""
        try:
            return self._coord_axes[self._output_axes].unit
        except AttributeError:
            return None

    @property
    def output_axes(self):
        """Return the output coordinate axes."""
        if self._coord_axes[self._output_axes] is not None:
            return self._coord_axes[self._output_axes]
        else:
            return self._output_axes

    @property
    def input_axes(self):
        """Return the input coordinate axes."""
        if self._coord_axes[self._input_axes] is not None:
            return self._coord_axes[self._input_axes]
        else:
            return self._input_axes

    @property
    def name(self):
        """Return the name for this WCS."""
        return self._name

    @name.setter
    def name(self, value):
        """Set the name for the WCS."""
        self._name = value

    def __str__(self):
        from astropy.table import Table
        col1 = [item[0] for item in self._pipeline]
        col2 = []
        for item in self._pipeline:
            model = item[1]
            if model is not None:
                if model.name != "":
                    col2.append(model.name)
                else:
                    col2.append(model.__class__.__name__)
            else:
                col2.append(None)
        t = Table([col1, col2], names=['From',  'Transform'])
        return str(t)

    def __repr__(self):
        fmt = "<WCS(output_axes={0}, input_axes={1}, forward_transform={2})>".format(
            self.output_axes, self.input_axes, self.forward_transform)
        return fmt

    def footprint(self, axes, center=True):
        """
        Return the footprint of the observation in world coordinates.

        Parameters
        ----------
        axes : tuple of floats
            size of image
        center : bool
            If `True` use the center of the pixel, otherwise use the corner.

        Returns
        -------
        coord : (4, 2) array of (*x*, *y*) coordinates.
            The order is counter-clockwise starting with the bottom left corner.
        """
        naxis1, naxis2 = axes  # extend this to more than 2 axes
        if center == True:
            corners = np.array([[1, 1],
                                [1, naxis2],
                                [naxis1, naxis2],
                                [naxis1, 1]], dtype=np.float64)
        else:
            corners = np.array([[0.5, 0.5],
                                [0.5, naxis2 + 0.5],
                                [naxis1 + 0.5, naxis2 + 0.5],
                                [naxis1 + 0.5, 0.5]], dtype=np.float64)
        result = self.__call__(corners[:, 0], corners[:, 1])
        return np.asarray(result).T
        #result = np.vstack(self.__call__(corners[:,0], corners[:,1])).T
        # try:
        # return self.output_coordinate_system.world_coordinates(result[:,0], result[:,1])
        # except:
        # return result
