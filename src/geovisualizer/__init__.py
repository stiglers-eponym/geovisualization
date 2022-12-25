"""
Geovisualizer: analyze and visualize GPX files
"""
__version__ = "0.0.1"

from geovisualizer.visualize import VisualizeAll
import matplotlib.pyplot as plt

def main(*files, mpl=True, pyqtgraph=True, separate_map=False, tile_visualization="surface", tile_zoom=11, zscale=2):
    gpx = VisualizeAll(*files, zscale=zscale)
    if pyqtgraph:
        gpx.plot_pyqtgraph(tile_zoom=tile_zoom, tile_visualization=tile_visualization)
    gpx.plot_mpl(bool(separate_map))
    gpx.print_all_statistics()
    if pyqtgraph:
        if mpl:
            plt.pause(2)
        gpx.plot_tracks()
        gpx.app.exec()
    if mpl:
        plt.show()
