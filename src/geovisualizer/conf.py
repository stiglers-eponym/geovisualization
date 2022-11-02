import os
CACHE_PATH = os.getenv("XDG_CACHE_HOME")
if CACHE_PATH is None:
    CACHE_PATH = os.path.join(os.getenv("HOME"), ".cache")

DEM_PATH = os.path.join(CACHE_PATH, "geodata/dem")
TILES_PATH = os.path.join(CACHE_PATH, "geodata/tiles")

VERTICAL_SPEED_SMOOTHEN_0 = 0.2
VERTICAL_SPEED_SMOOTHEN_1 = 0.1
VERTICAL_SPEED_SMOOTHEN_2 = 0.01
HORIZONTAL_SPEED_SMOOTHEN = 0.5
POWER_SMOOTHEN_1 = 0.1
POWER_SMOOTHEN_2 = 0.01
GRADIENT_SMOOTHEN = 0.1
GRADIENT_NORM = 0.1
VSPEED_NORM = 1.
