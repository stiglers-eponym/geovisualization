import libxml2
import numpy as np


def read_gpx_file(filename):
    file = libxml2.readFile(filename, "utf-8", 0)
    element = file.children
    if element.name != "gpx":
        raise ValueError("File not understood")
    element = element.children
    while element and element.name != "trk":
        element = element.next
    if element is None:
        raise ValueError("File not understood")
    element = element.children
    segments = []
    while element:
        if element.name == "trkseg":
            trackpoints = []
            trkpt = element.children
            while trkpt:
                if trkpt.type == "element" and trkpt.name == "trkpt":
                    trackpoints.append(read_trackpoint(trkpt))
                trkpt = trkpt.next
            segments.append(trackpoints)
        element = element.next

    def get_arr(name):
        return [np.array([np.float64(pt.get(name, "NaN")) for pt in trackpoints]) \
                for trackpoints in segments]

    return dict(
        lat = get_arr("lat"),
        lon = get_arr("lon"),
        elevation = get_arr("ele"),
        hdop = get_arr("hdop"),
        speed = get_arr("speed"),
        time = [np.array([np.datetime64(pt.get("time", "")) for pt in trackpoints]) \
                for trackpoints in segments],
        )


def read_trackpoint(trkpt):
    assert trkpt.type == "element" and trkpt.name == "trkpt"
    result = {}
    prop = trkpt.properties
    while prop:
        if prop.type == "attribute":
            result[prop.name] = prop.content
        prop = prop.next
    child = trkpt.children
    while child and child.name != "extensions":
        if child.type == "element":
            result[child.name] = child.content
        child = child.next
    if child and child.name == "extensions":
        child = child.children
        while child:
            if child.type == "element":
                result[child.name] = child.content
            child = child.next
    return result
