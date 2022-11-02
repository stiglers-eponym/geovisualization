#!/usr/bin/env python
from geovisualizer import main
import argparse

def parse():
    parser = argparse.ArgumentParser(description="Basic GPX analyzer")
    parser.add_argument("--mpl", type=int, metavar="0 or 1", default=1, help="show matplotlib plot")
    parser.add_argument("--separate_map", type=int, metavar="0 or 1", default=0, help="show matplotlib map in separate window")
    parser.add_argument("--pyqtgraph", type=int, metavar="0 or 1", default=1, help="show PyQtGraph 3d plot")
    parser.add_argument("--zoom", type=int, metavar="int", default=12, help="zoom level for elevation tiles")
    parser.add_argument("--tile_visualization", type=str, choices=("surface", "scatter"), default="surface", help="elevation visualization")
    parser.add_argument("files", type=str, nargs="+", help="GPX files")
    args = parser.parse_args()
    main(*args.files, mpl=args.mpl, pyqtgraph=args.pyqtgraph, separate_map=args.separate_map, tile_zoom=args.zoom, tile_visualization=args.tile_visualization)


if __name__ == "__main__":
    parse()
