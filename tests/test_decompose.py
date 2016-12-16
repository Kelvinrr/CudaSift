import numpy as np
import cudasift as cs
from plio.io import io_gdal

cs.PyInitCuda()
img1 = io_gdal.GeoDataset('/data/autocnet/autocnet/examples/Apollo15/AS15-M-0295_SML.png')
img2 = io_gdal.GeoDataset('/data/autocnet/autocnet/examples/Apollo15/AS15-M-0296_SML.png')

arr1 = img1.read_array()
arr2 = img2.read_array()

mem1, mem2 = cs.PyDecomposeAndMatch(arr1, arr2)
