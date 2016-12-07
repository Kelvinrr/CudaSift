# -*- coding: utf8 -*-
'''Python wrapper for CudaSift

Please cite
M. Björkman, N. Bergström and D. Kragic, "Detecting, segmenting and tracking
unknown objects using multi-label MRF inference",
CVIU, 118, pp. 111-127, January 2014.
'''
import pyximport; pyximport.install()

from ._cudasift import PyInitCuda, PySiftData, ExtractKeypoints, PyMatchSiftData, PyRadialMean

all = [PyInitCuda, PySiftData, ExtractKeypoints, PyMatchSiftData, PyRadialMean]
