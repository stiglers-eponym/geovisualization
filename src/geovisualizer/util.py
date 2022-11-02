"""
Some functions for dealing with different data formats and coordinate
systems.
"""
import os
from numbers import Number
import numpy as np
from PIL import Image
from pyproj import Proj, transform
from geovisualizer import conf

R_EARTH = 6371.0
R_EARTH_POLE = 6365.752314
R_EARTH_EQUATOR = 6378.137


def smoothen(array, alpha, weights=None):
    assert alpha > 0
    nmax = (10/alpha)**0.5
    gauss = np.exp(-alpha*np.arange(-nmax, nmax+1)**2)
    gauss /= gauss.sum()
    if np.isfinite(array).all():
        if weights is None:
            return np.convolve(array, gauss, "same")
        return np.convolve(weights*array, gauss, "same")/np.convolve(weights, gauss, "same")
    result = np.empty_like(array)
    good = np.isfinite(array)
    result[~good] = np.nan
    if weights is None:
        result[good] = np.convolve(array[good], gauss, "same")
    else:
        result[good] = np.convolve(weights[good]*array[good], gauss, "same")/np.convolve(weights[good], gauss, "same")
    return result


def transform_xy_lonlat(x, y, init="epsg:25832"):
    """
    transform x, y input data to longitude, lattitude using pyproj
    """
    inProj = Proj(init=init)
    outProj = Proj(init="epsg:4326")
    return transform(inProj, outProj, x, y)


def lonlat_to_xytiles(zoom, lon, lat):
    n = 1 << zoom
    xtile = n * (lon/np.pi + 1)/2
    ytile = n * (1 - np.log(np.tan(lat) + 1/np.cos(lat))/np.pi)/2
    return int(xtile), int(ytile)


def read_terrarium_rgb(filename):
    data = np.asarray(Image.open(filename).convert("RGB"))
    return (data[...,0] * 256 + data[...,1] + data[...,2] / 256) - 32768


def read_terrarium_tile(zoom, xtile, ytile):
    z = read_terrarium_rgb(os.path.join(conf.TILES_PATH, f"terrarium_{zoom}_{xtile}_{ytile}.png"))
    n = 1 << zoom
    xtile_arr = np.linspace(xtile, xtile+1, 256, endpoint=False)
    ytile_arr = np.linspace(ytile, ytile+1, 256, endpoint=False)
    lon = np.pi*(2*xtile_arr/n - 1)
    lat = np.arctan(np.sinh(np.pi*(1-2*ytile_arr/n)))
    return lon, lat, z


def segment_lengths(lon, lat):
    lat_segments = (lat[1:]+lat[:-1])/2
    r_earth = np.sqrt((np.cos(lat_segments)*R_EARTH_EQUATOR)**2 + (np.sin(lat_segments)*R_EARTH_POLE)**2)
    x_diff = r_earth * (lat[1:] - lat[:-1])
    y_diff = r_earth * np.cos(lat_segments) * (lon[1:]-lon[:-1])
    return np.sqrt(x_diff**2+y_diff**2)


def coordinate_transform(lon, lat, ref_lon, ref_lat):
    """
    Transform spherical coordinates (lat, lon) by setting
    (ref_lat, ref_lon) as new (0, 0) coordinate.
    All angles are in radiants.
    """
    lon = lon - ref_lon
    normal = np.array([-np.sin(ref_lat), 0, np.cos(ref_lat)])
    xyz = np.array([np.cos(lon)*np.cos(lat), np.sin(lon)*np.cos(lat), np.sin(lat)])
    new_lat = np.arcsin(np.tensordot(normal, xyz, (0,0)))
    if isinstance(lon, Number):
        if np.abs(lon) <= np.pi/4 or np.cos(lat)/np.cos(new_lat) <= 0.5:
            new_lon = np.arcsin(xyz[1]/np.cos(new_lat))
        elif np.abs(lon) <= 3*np.pi/4:
            new_lon = np.sign(lon)*np.arccos(xyz[0]/np.cos(new_lat))
        else:
            new_lon = np.sign(lon)*np.pi - np.arcsin(xyz[1]/np.cos(new_lat))
    else:
        new_lon = np.arcsin(xyz[1]/np.cos(new_lat))
        if np.abs(lon).max() > np.pi/4:
            sel = (np.abs(lon)>np.pi/4) & (np.abs(lon)<=3*np.pi/4)
            new_lon[sel] = np.sign(lon[sel])*np.arccos(xyz[0][sel]/np.cos(new_lat))
            if np.abs(lon).max() > 3*np.pi/4:
                new_lon[lon>3*np.pi/4] = np.pi - new_lon[lon>3*np.pi/4]
                new_lon[lon<-3*np.pi/4] = -np.pi - new_lon[lon<-3*np.pi/4]
    return new_lon, new_lat


def extend_array(arr):
    new_arr = np.empty(shape=(arr.shape[0]+1, *arr.shape[1:]), dtype=arr.dtype)
    new_arr[1:-1] = (arr[1:] + arr[:-1])/2
    new_arr[0] = arr[0]
    new_arr[-1] = arr[-1]
    return new_arr
