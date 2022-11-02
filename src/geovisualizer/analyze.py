import numpy as np
from geovisualizer.gpx import read_gpx_file
from geovisualizer.util import *
from geovisualizer import conf

G_CONSTANT = 9.80665


class GpxAnalyze:
    def __init__(self, filename):
        self.filename = filename
        properties = read_gpx_file(filename)
        for key, value in properties.items():
            setattr(self, key, np.concatenate(value))
        self.lon *= np.pi/180
        self.lat *= np.pi/180

        self.segment_lengths = segment_lengths(self.lon, self.lat)
        self.cum_distance = np.append(0, np.cumsum(self.segment_lengths))

        if np.isfinite(self.time).any():
            self.rel_time = (self.time - self.time[0]).astype(np.float64)
            time_diff = self.rel_time[1:] - self.rel_time[:-1]
            weights = time_diff
            self.has_time = True
        else:
            self.has_time = False
            weights = self.segment_lengths

        self.gradient = 1e-3 * (self.elevation[1:] - self.elevation[:-1]) / self.segment_lengths
        self.gradient_smooth = smoothen(self.gradient, conf.GRADIENT_SMOOTHEN, weights)

        if self.has_time:
            self.vertical_speed = (self.elevation[1:] - self.elevation[:-1]) / time_diff
            self.vertical_speed_smooth = smoothen(self.vertical_speed,
                                                  conf.VERTICAL_SPEED_SMOOTHEN_1,
                                                  weights)
            self.vertical_speed_ssmooth = smoothen(self.vertical_speed,
                                                   conf.VERTICAL_SPEED_SMOOTHEN_2,
                                                   weights)

            self.alt_speed = 1e3 * self.segment_lengths / time_diff
            self.alt_speed_smooth = smoothen(self.alt_speed, 0.5, weights)

            nan_speed = np.nan_to_num(self.speed, nan=0.0)
            self.power = G_CONSTANT*self.vertical_speed + 0.5*(nan_speed[1:]**2 - nan_speed[:-1]**2) / time_diff
            self.power_noisy = G_CONSTANT*smoothen(self.vertical_speed, conf.VERTICAL_SPEED_SMOOTHEN_0, weights) \
                    + 0.5*smoothen((nan_speed[1:]**2 - nan_speed[:-1]**2) / time_diff, conf.HORIZONTAL_SPEED_SMOOTHEN, weights)
            self.power_smooth = smoothen(self.power, conf.POWER_SMOOTHEN_1, weights)
            self.power_ssmooth = smoothen(self.power, conf.POWER_SMOOTHEN_2, weights)


    def print_statistics(self):
        distance = self.segment_lengths.sum()
        #simple_distance = (((self.x[1:]-self.x[:-1])**2 + (self.y[1:]-self.y[:-1])**2)**0.5).sum()
        print(f"distance: {distance:.4g}km")
        if not self.has_time:
            return
        duration = self.rel_time[-1]
        avg_speed = 1e3*distance/duration
        max_speed = max(np.nanmax(self.speed), np.nanmax(self.alt_speed_smooth))
        max_power = np.nanmax(self.power_smooth)
        max_power_smooth = np.nanmax(self.power_ssmooth)
        time_diff = self.rel_time[1:] - self.rel_time[:-1]
        ele_diff = smoothen(self.elevation[1:] - self.elevation[:-1], 0.5, time_diff)
        total_climb = ele_diff[ele_diff>0].sum()
        ele_diff_smooth = smoothen(self.elevation[1:] - self.elevation[:-1], 0.2, time_diff)
        total_climb_smooth = ele_diff_smooth[ele_diff_smooth>0].sum()
        total_climb_duration = time_diff[self.power_smooth>1].sum()
        avg_climb_power = np.nanmean(self.power_ssmooth[self.power_ssmooth>1])
        max_break_power = self.power_noisy.min()
        print(f"duration: {duration/60:.5g}min")
        print(f"avg. speed: {3.6*avg_speed:.4g}km/h")
        print(f"avg. climb power: {avg_climb_power:.3g}W/kg in {total_climb_duration/60:.3g}min")
        print(f"max. climb power: {max_power:.3g}W/kg or smoothened {max_power_smooth:.3g}W/kg")
        print(f"max. break power: {max_break_power:.3g}W/kg")
        print(f"total climb: {total_climb:.4g}m or smoothened {total_climb_smooth:.4g}m")
