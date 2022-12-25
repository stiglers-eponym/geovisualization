import os
import re
import numpy as np
from pyqtgraph.Qt.QtWidgets import QApplication
import pyqtgraph.opengl as gl
import matplotlib.pyplot as plt
from geovisualizer.util import *
from geovisualizer.gpx import read_gpx_file
from geovisualizer.analyze import GpxAnalyze
from geovisualizer import conf


class VisualizeGpx:
    def __init__(self, analyzer:GpxAnalyze, ref_lon, ref_lat, separate_map=False):
        self.analyzer = analyzer
        self.separate_map = separate_map
        transformed_lon, transformed_lat = coordinate_transform(analyzer.lon, analyzer.lat, ref_lon, ref_lat)
        r_earth = np.sqrt((np.cos(ref_lat)*R_EARTH_EQUATOR)**2 + (np.sin(ref_lat)*R_EARTH_POLE)**2)
        self.x = r_earth*transformed_lon
        self.y = r_earth*transformed_lat
        self.update_position = lambda lon, lat, elev: None

        self.create_axes()
        self.plot_track("time" if analyzer.has_time else "distance")
        self.plot_map()
        self.add_interactive()


    def create_axes(self, xaxis="time"):
        if self.separate_map:
            if self.analyzer.has_time:
                self.fig, (self.ax_vert_speed, self.ax_elev) = plt.subplots(
                        nrows=2,
                        ncols=1,
                        num=self.analyzer.filename)
                self.fig.subplots_adjust(left=0.08, bottom=0.08, right=0.93, top=0.93)
                self.fig_map, self.ax_map = plt.subplots(num="map: " + self.analyzer.filename)
                self.fig_map.subplots_adjust(left=0.06, bottom=0.06, right=1, top=0.95)
            else:
                self.fig, self.ax_elev = plt.subplots(nrows=1, num=self.analyzer.filename)
                self.fig.subplots_adjust(left=0.08, bottom=0.08, right=0.93, top=0.93)
                self.fig_map, self.ax_map = plt.subplots(num="map: " + self.analyzer.filename)
                self.fig_map.subplots_adjust(left=0.06, bottom=0.06, right=1, top=0.95)
                self.ax_vert_speed = None
        else:
            if self.analyzer.has_time:
                self.fig, (self.ax_vert_speed, self.ax_elev, self.ax_map) = plt.subplots(
                        nrows=3,
                        ncols=1,
                        num=self.analyzer.filename)
                self.fig.subplots_adjust(left=0.08, bottom=0.05, right=0.93, top=0.96, hspace=0.2)
            else:
                self.fig, (self.ax_elev, self.ax_map) = plt.subplots(nrows=2, num=self.analyzer.filename)
                self.fig.subplots_adjust(left=0.08, bottom=0.08, right=0.93, top=0.93)
                self.ax_vert_speed = None
        if self.ax_vert_speed is not None:
            self.ax_power = self.ax_vert_speed.twinx()
            self.ax_elev.sharex(self.ax_vert_speed)
            self.ax_vert_speed.set_ylabel("cm/s vertical")
            self.ax_power.set_ylabel("W/kg")
            self.ax_power.set_title("power / vertical speed")
        else:
            self.ax_power = None
        self.ax_speed = self.ax_elev.twinx()
        self.ax_speed.set_ylabel("km/h")
        self.ax_elev.set_title("elevation / speed")
        self.ax_elev.set_ylabel("m above sea")
        self.ax_elev.set_xlabel("min")
        self.ax_map.set_title("map")
        self.ax_map.set_xlabel("km")
        self.ax_map.set_ylabel("km")
        self.ax_map.set_aspect(1)


    def plot_track(self, xaxis="time"):
        match xaxis:
            case "time":
                self.xaxis_nodes = self.analyzer.rel_time/60
            case "distance":
                self.xaxis_nodes = self.analyzer.cum_distance
            case _:
                raise ValueError(f"invalid value for xaxis: {xaxis}")
        self.xaxis_edges = (self.xaxis_nodes[1:] + self.xaxis_nodes[:-1])/2
        if self.ax_vert_speed is not None:
            self.ax_vert_speed.plot(
                    self.xaxis_edges,
                    100*self.analyzer.vertical_speed_smooth,
                    color="blue",
                    linewidth=1)
            self.ax_vert_speed.plot(
                    self.xaxis_edges,
                    100*self.analyzer.vertical_speed_ssmooth,
                    color="green",
                    linewidth=2)
        if self.ax_power is not None:
            self.ax_power.fill_between(
                    self.xaxis_edges,
                    self.analyzer.power_ssmooth,
                    color="red",
                    alpha=0.2,
                    zorder=-2)
            self.ax_power.plot(
                    self.xaxis_edges,
                    self.analyzer.power_noisy,
                    ".",
                    color="black",
                    markersize=2)
            self.ax_power.plot(
                    self.xaxis_edges,
                    self.analyzer.power_smooth,
                    color="black",
                    linewidth=1)
            self.ax_power.plot(
                    self.xaxis_edges,
                    self.analyzer.power_ssmooth,
                    color="red",
                    linewidth=2)

        self.ax_elev.plot(self.xaxis_nodes, self.analyzer.elevation, "blue")
        if self.analyzer.has_time:
            self.ax_speed.plot(self.xaxis_edges, 3.6*self.analyzer.alt_speed_smooth, "green")
        try:
            self.ax_speed.plot(self.xaxis_nodes, 3.6*self.analyzer.speed, "red")
        except AttributeError:
            pass


    def plot_map(self):
        self.ax_map.set_facecolor("#404040")
        self.ax_map.plot(self.x, self.y, color="white", zorder=-2, linewidth=3.2)
        if self.analyzer.has_time:
            self.ax_map.scatter(
                    (self.x[1:]+self.x[:-1])/2,
                    (self.y[1:]+self.y[:-1])/2,
                    c=self.analyzer.vertical_speed_smooth,
                    cmap=plt.cm.seismic,
                    norm=plt.Normalize(-conf.VSPEED_NORM, conf.VSPEED_NORM),
                    s=2,
                    zorder=-1)
        else:
            self.ax_map.scatter(
                    (self.x[1:]+self.x[:-1])/2,
                    (self.y[1:]+self.y[:-1])/2,
                    c=self.analyzer.gradient_smooth,
                    cmap=plt.cm.seismic,
                    norm=plt.Normalize(-conf.GRADIENT_NORM, conf.GRADIENT_NORM),
                    s=2,
                    zorder=-1)
        self.map_pos, = self.ax_map.plot(np.nan, np.nan, "or")


    def move_to(self, x):
        if x is None:
            self.move_to_none()
            return
        node_index = self.xaxis_nodes.searchsorted(x)
        if node_index > self.xaxis_nodes.size:
            self.move_to_none()
            return
        edge_index = node_index - (self.xaxis_edges[node_index-1] < x)
        try:
            self.vline_power_ax.set_xdata([x, x])
        except AttributeError:
            pass
        try:
            self.vline_elev_ax.set_xdata([x, x])
        except AttributeError:
            pass
        if self.analyzer.has_time:
            self.ax_power.set_title(f"power: {self.analyzer.power_smooth[edge_index]:.3g}W/kg or {self.analyzer.power_ssmooth[edge_index]:.3g}W/kg (smoothened), gradient: {100*self.analyzer.gradient_smooth[edge_index]:.3g}%")
            self.ax_elev.set_title(f"elevation: {self.analyzer.elevation[node_index]:.4g}m, speed: {3.6*self.analyzer.speed[node_index]:.3g}km/h or {3.6*self.analyzer.alt_speed_smooth[edge_index]:.3g}km/h (GPS position)")
        else:
            self.ax_elev.set_title(f"elevation: {self.analyzer.elevation[node_index]:.4g}m")
        self.ax_map.set_title(f"map: {self.analyzer.lat[node_index]*180/np.pi:.8f}°, {self.analyzer.lon[node_index]*180/np.pi:.8f}°")
        try:
            self.map_pos.set_data(self.x[node_index], self.y[node_index])
        except AttributeError:
            pass
        self.fig.canvas.draw_idle()
        try:
            self.fig_map.canvas.draw_idle()
        except AttributeError:
            pass
        self.update_position(self.analyzer.lon[node_index], self.analyzer.lat[node_index], self.analyzer.elevation[node_index])


    def move_to_none(self):
        try:
            self.vline_power_ax.set_xdata([np.nan, np.nan])
        except AttributeError:
            pass
        try:
            self.vline_elev_ax.set_xdata([np.nan, np.nan])
        except AttributeError:
            pass
        self.point_on_map.set_data(np.nan, np.nan)
        if self.ax_power is not None:
            self.ax_power.set_title("power / vertical speed")
        self.ax_elev.set_title("elevation / speed")
        self.ax_map.set_title("map")
        self.fig.canvas.draw_idle()
        try:
            self.fig_map.canvas.draw_idle()
        except AttributeError:
            pass
        self.update_position(np.nan, np.nan, np.nan)


    def add_interactive(self):
        if self.ax_power is not None:
            self.vline_power_ax, = self.ax_power.plot(
                    [np.nan, np.nan],
                    [0, 1],
                    transform=self.ax_power.get_xaxis_transform())
        self.vline_elev_ax, = self.ax_elev.plot(
                [np.nan, np.nan],
                [0, 1],
                transform=self.ax_elev.get_xaxis_transform())
        self.point_on_map, = self.ax_map.plot(
                np.nan, np.nan, 'o', color="red", alpha=0.8)

        def move(event):
            if event.inaxes in (self.ax_power, self.ax_vert_speed, self.ax_elev, self.ax_speed):
                self.move_to(event.xdata)
            else:
                self.move_to_none()
        self.fig.canvas.mpl_connect("motion_notify_event", move)




class VisualizeAll:
    def __init__(self, *files, zscale=2):
        self.files = files
        self.analyzers = [GpxAnalyze(file) for file in files]
        lat_min = min(np.nanmin(a.lat) for a in self.analyzers)
        lat_max = max(np.nanmax(a.lat) for a in self.analyzers)
        lon_min = min(np.nanmin(a.lon) for a in self.analyzers)
        lon_max = max(np.nanmax(a.lon) for a in self.analyzers)
        self.extent = [lon_min, lon_max, lat_min, lat_max]
        self.ref_lat = (lat_min + lat_max)/2
        self.ref_lon = (lon_min + lon_max)/2
        self.gpx_visualizers = []
        self.z_baseline = np.nanmin([np.nanmin(a.elevation) for a in self.analyzers]) - 10
        self.z_scale = zscale
        self.z_max = 1.1*np.nanmax([np.nanmax(a.elevation) for a in self.analyzers])


    def print_all_statistics(self):
        for analyzer in self.analyzers:
            print(analyzer.filename)
            analyzer.print_statistics()
            print()


    def plot_pyqtgraph(self, tile_zoom=11, tile_visualization="scatter"):
        self.app = QApplication([])
        self.widget = gl.GLViewWidget()
        self.widget.show()
        self.widget.setWindowTitle("GPX visualizer")
        self.zgrid = gl.GLGridItem()
        self.widget.addItem(self.zgrid)

        self.track_scatter_plots = []
        self.tile_scatter_plots = []
        self.tile_surf_plots = []

        self.sp_pos = gl.GLScatterPlotItem(pos=np.array([[0.,0.,0.]]), color=(1,0,0,0.8), size=0.2, pxMode=False)
        self.sp_pos.setGLOptions("translucent")
        self.widget.addItem(self.sp_pos)

        if tile_visualization:
            self.pyqtgraph_add_all_tiles(zoom=tile_zoom, visualization=tile_visualization)


    def plot_tracks(self):
        for v in self.gpx_visualizers:
            self.plot_track(v)


    def plot_mpl(self, separate_map=False):
        def update(*args):
            self.update_position(*args)
        for a in self.analyzers:
            v = VisualizeGpx(a, self.ref_lon, self.ref_lat, separate_map=separate_map)
            v.update_position = update
            self.gpx_visualizers.append(v)


    def update_position(self, lon, lat, elev):
        try:
            self.sp_pos.setData(pos=np.array([[*self.to_xyz(lon, lat, elev)]]))
        except AttributeError:
            pass


    def to_xyz(self, lon, lat, elev):
        lon_transformed, lat_transformed = coordinate_transform(
                lon, lat,
                self.ref_lon, self.ref_lat)
        r_earth = np.sqrt((np.cos(self.ref_lat)*R_EARTH_EQUATOR)**2 + (np.sin(self.ref_lat)*R_EARTH_POLE)**2)
        return r_earth*lon_transformed, r_earth*lat_transformed, self.z_scale/1000*(elev - self.z_baseline)


    def plot_track(self, visualizer):
        pos_nodes = np.array((self.to_xyz(visualizer.analyzer.lon, visualizer.analyzer.lat, visualizer.analyzer.elevation))).T
        grad = (extend_array(visualizer.analyzer.gradient_smooth)/conf.GRADIENT_NORM + 1)/2
        sp_gradient = gl.GLScatterPlotItem(pos=pos_nodes, color=plt.cm.seismic(grad), size=0.07, pxMode=False)
        sp_gradient.setGLOptions("translucent")
        self.widget.addItem(sp_gradient)
        self.track_scatter_plots.append(sp_gradient)


    def pyqtgraph_add_geodata(self):
        """
        Experimental!
        """
        raw_data = []
        for file in os.listdir(conf.DEM_PATH):
            if file[-7:] != ".xyz.gz":
                continue
            raw_data.append(np.genfromtxt(os.path.join(conf.DEM_PATH, file)))
        array = np.concatenate(raw_data)
        del raw_data
        array[:,0], array[:,1] = self.lonlat_deg_to_xy(*transform_xy_lonlat(array[:,0], array[:,1]))
        array[:,2] -= self.ref_elevation
        array[:,2] *= self.z_scale/1000
        z_normalized = array[:,2] - self.z_baseline
        z_normalized /= np.nanmax(z_normalized)
        self.sp_geodata = gl.GLScatterPlotItem(pos=array, color=plt.cm.viridis(z_normalized), size=0.01, pxMode=True)
        self.widget.addItem(self.sp_geodata)


    def pyqtgraph_add_elevation_tile_at(self, lon, lat, zoom=11, visualization="surface"):
        self.pyqtgraph_add_elevation_tile(*lonlat_to_xytiles(zoom, lon, lat), zoom, visualization=visualization)


    def pyqtgraph_add_elevation_tile(self, xtile, ytile, zoom=11, visualization="surface"):
        lon, lat, z = read_terrarium_tile(zoom, xtile, ytile)
        x, y, z = self.to_xyz(*np.meshgrid(lon, lat), z)
        z_normalized = 1e3 / ((self.z_max - self.z_baseline) * self.z_scale) * z
        match visualization:
            case "surface":
                z_normalized = 1e3 / ((self.z_max - self.z_baseline) * self.z_scale) * z
                surfplot_tile = gl.GLSurfacePlotItem(x=x[x.shape[0]//2,:], y=y[:,y.shape[1]//2], z=z.T-0.02, colors=plt.cm.viridis(z_normalized.T))
                self.widget.addItem(surfplot_tile)
                self.tile_surf_plots.append(surfplot_tile)
            case "scatter":
                pos = np.array([x, y, z]).transpose((1,2,0)).reshape((-1,3))
                scatterplot_tile = gl.GLScatterPlotItem(pos=pos, color=plt.cm.viridis(z_normalized.flat), pxMode=False, size=0.02)
                self.widget.addItem(scatterplot_tile)
                self.tile_scatter_plots.append(scatterplot_tile)


    def pyqtgraph_add_all_tiles(self, zoom=11, visualization="surface"):
        for filename in os.listdir(conf.TILES_PATH):
            match = re.match(f"terrarium_{zoom}_([0-9]+)_([0-9]+)\\.png$", filename)
            if match is None:
                continue
            xtile = int(match.group(1))
            ytile = int(match.group(2))
            n = 1 << zoom
            lon_min = np.pi*(2*xtile/n - 1)
            lon_max = np.pi*(2*(xtile+255/256)/n - 1)
            lat_max = np.arctan(np.sinh(np.pi*(1-2*ytile/n)))
            lat_min = np.arctan(np.sinh(np.pi*(1-2*(ytile+255/256)/n)))
            if lon_max >= self.extent[0] and lon_min <= self.extent[1] and lat_max >= self.extent[2] and lat_min <= self.extent[3]:
                self.pyqtgraph_add_elevation_tile(xtile, ytile, zoom, visualization=visualization)
