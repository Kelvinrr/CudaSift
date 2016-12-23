import ctypes
import numpy as np
import pandas

from scipy.spatial.distance import cdist

from libc.stdint cimport uintptr_t
from libc.string cimport memcpy
from libc.stdio cimport printf
from libcpp cimport bool

cimport numpy as np

cdef extern from "device_types.h":
    ctypedef enum cudaMemcpyKind:
        cudaMemcpyHostToHost,
        cudaMemcpyHostToDevice,
        cudaMemcpyDeviceToHost,
        cudaMemcpyDeviceToDevice,
        cudaMemcpyDefault

cdef extern from "cuda_runtime_api.h" nogil:
    ctypedef int cudaError_t
    ctypedef int cudaMemoryType
    cdef struct cudaPointerAttributes:
        cudaMemoryType memoryType
        int device
        void *devicePointer
        void *hostPointer
        int isManaged
    cdef cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *attributes, void *ptr) nogil
    cdef cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, int kind)
    cdef cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags)
    cdef cudaError_t cudaHostUnregister(void *ptr)
    cdef cudaError_t cudaMallocHost(void *ptr, size_t size)
    cdef cudaError_t cudaFreeHost(void *ptr)
    cdef cudaError_t cudaMemset(void *dst, int value, size_t count)

cdef extern from "cudaImage.h" nogil:
    cdef cppclass CudaImage:
      int width
      int height
      int pitch
      float *h_data
      float *d_data
      float *t_data
      bool d_internalAlloc
      bool h_internalAlloc

      void Allocate(int width, int height, int pitch, bool withHost, float *devMem, float *hostMem)
      double Download()
      double Readback()
      double InitTexture()
      double CopyToTexture(CudaImage &dst, bool host)

    cdef int iDivUp(int a, int b)
    cdef int iDivDown(int a, int b)
    cdef int iAlignUp(int a, int b)
    cdef int iAlignDown(int a, int b)

cdef extern from "cudaDecompose.h" nogil:
    cdef void RadialMean(int steps, int h, int w, float *image, float *classified, float *means)
    cdef void DecomposeAndMatch(CudaImage &img1, CudaImage &img2,
                                CudaImage &mem1, CudaImage &mem2,
                                int soriginx, int soriginy, int doriginx, int doriginy,
                                int start,
                                int *source_extent, int *destination_extent)

cdef extern from "cudaSift.h" nogil:
    ctypedef struct SiftPoint:
        float xpos
        float ypos
        float scale
        float sharpness
        float edgeness
        float orientation
        float score
        float ambiguity
        int match
        float match_xpos
        float match_ypos
        float match_error
        float subsampling
        float empty[3]
        float data[128]

    ctypedef struct SiftData:
        int numPts
        int maxPts
        SiftPoint *h_data
        SiftPoint *d_data
    cdef void InitCuda(int devNum)
    cdef void ExtractSift(
        SiftData &siftData, CudaImage &img, int numOctaves,
        double initBlur, float thresh, float lowestScale,
        float subsampling)
    cdef void InitSiftData(SiftData &data, int num, bool host, bool dev)
    cdef void FreeSiftData(SiftData &data)
    cdef void PrintSiftData(SiftData &data)
    cdef double MatchSiftData(SiftData &data1, SiftData &data2)
    cdef double FindHomography(
        SiftData &data,  float *homography, int *numMatches,
        int numLoops, float minScore, float maxAmbiguity,
        float thresh)

def PyRadialMean(image, classified):
    cdef:
      size_t w = classified.shape[1]
      size_t h = classified.shape[0]
      int steps = 720
      np.ndarray tmpi = np.ascontiguousarray(image.astype(np.float32))
      void *pSrc = tmpi.data

      np.ndarray tmpc = np.ascontiguousarray(classified.astype(np.float32))
      void *pClass = tmpc.data

      np.ndarray means = np.ascontiguousarray(np.empty(steps, dtype=np.float32))
      void *pMeans = means.data

    with nogil:
      RadialMean(steps, h, w, <float *>pSrc, <float *>pClass, <float *>pMeans)
    return means

def PyDecomposeAndMatch(img1, img2, maxiterations=3, ratio=0.9, buf_dist=3):
    cdef:
        # Get the image data allocated
        CudaImage cimg1
        CudaImage cimg2
        np.ndarray tmp1 = np.ascontiguousarray(img1.astype(np.float32))
        void *pImg1 = tmp1.data
        size_t w1 = img1.shape[1]
        size_t h1 = img1.shape[0]
        np.ndarray tmp2 = np.ascontiguousarray(img2.astype(np.float32))
        void *pImg2 = tmp2.data
        size_t w2 = img2.shape[1]
        size_t h2 = img2.shape[0]

        int source_extent[4]
        int destination_extent[4]

        # Get the membership data allocated
        CudaImage cmem1
        CudaImage cmem2
        np.ndarray mem1 = np.zeros(img1.shape, dtype=np.float32)
        np.ndarray mem2 = np.zeros(img2.shape, dtype=np.float32)
        #np.ndarray tcmem1 = np.ascontiguousarray(mem1.astype(np.float32))
        void *pMem1 = mem1.data
        #np.ndarray tcmem2 = np.ascontiguousarray(mem2.astype(np.float32))
        void *pMem2 = mem2.data

        # Get the SIFT data setup
        # Parameters for the SIFT extraction
        int numOctaves = 5,
        float initBlur = 0,
        float thresh = 5,
        float lowestScale = 0,
        float subsampling = 1.0

    # Download the two images to the GPU
    with nogil:
        # Allocate and download the images
        cimg1.Allocate(w1, h1, iAlignUp(w1, 128), False, NULL, <float *>pImg1)
        cimg2.Allocate(w2, h2, iAlignUp(w2, 128), False, NULL, <float *>pImg2)
        cimg1.Download()
        cimg2.Download()

        # Allocate and download the memberships
        cmem1.Allocate(w1, h1, iAlignUp(w1, 128), False, NULL, <float *>pMem1)
        cmem2.Allocate(w1, h1, iAlignUp(w2, 128), False, NULL, <float *>pMem2)
        cmem1.Download()
        cmem2.Download()

    # Instantiate the PySiftDAta
    sd1 = PySiftData()
    sd2 = PySiftData()

    del tmp1
    del tmp2

    # Now extract the keypoints
    with nogil:
        ExtractSift(sd1.data, cimg1, numOctaves, initBlur, thresh,
                        lowestScale, subsampling)
        ExtractSift(sd2.data, cimg2, numOctaves, initBlur, thresh,
                        lowestScale, subsampling)

    # Grab the keypoints that have been extracted
    source_keypoints, source_descriptors = sd1.to_data_frame()
    destination_keypoints, destination_descriptors = sd2.to_data_frame()

    start = 1
    for i in range(maxiterations):
        # Read the membership information back from the GPU
        # This should update the CudaImage.h_data attribute
        cmem1.Readback()
        cmem2.Readback()
        # Step over the unique membership values
        for p in np.unique(mem1):
            print('Processing Decomposition: ', p)
            sy_part, sx_part = np.where(mem1 == p)
            dy_part, dx_part = np.where(mem2 == p)

            # Get the source extent
            minsy = np.min(sy_part)
            maxsy = np.max(sy_part) + 1
            minsx = np.min(sx_part)
            maxsx = np.max(sx_part) + 1
            source_extent[:] = minsy, maxsy, minsx, maxsx
            # Get the destination extent
            mindy = np.min(dy_part)
            maxdy = np.max(dy_part) + 1
            mindx = np.min(dx_part)
            maxdx = np.max(dx_part) + 1
            destination_extent[:] = mindy, maxdy, mindx, maxdx

            scounter = 0
            decompose = False
            while True:
                # Get the source and destination keypoints that are in the current subregion
                sub_source_keypoints = source_keypoints.query('xpos >= {} and xpos <= {} and ypos >= {} and ypos <= {}'.format(minsx, maxsx, minsy, maxsy))
                if len(sub_source_keypoints) == 0:
                    break  # No valid keypoints in this (sub)image

                sub_destination_keypoints = destination_keypoints.query('xpos >= {} and xpos <= {} and ypos >= {} and ypos <= {}'.format(mindx, maxdx, mindy, maxdy))
                if len(sub_destination_keypoints) == 0:
                    break  # No valid keypoints in this (sub)image

                sub_skps = sub_source_keypoints[['xpos', 'ypos', 'scale', 'sharpness', 'edgeness', 'orientation', 'score', 'ambiguity']]
                sub_dkps = sub_destination_keypoints[['xpos', 'ypos', 'scale', 'sharpness', 'edgeness', 'orientation', 'score', 'ambiguity']]
                # Create temporary SiftData objects and match them

                tsd1 = PySiftData.from_data_frame(sub_skps, source_descriptors[sub_source_keypoints.index])
                tsd2 = PySiftData.from_data_frame(sub_dkps, destination_descriptors[sub_destination_keypoints.index])

                PyMatchSiftData(tsd1, tsd2)
                localmatches, _ = tsd1.to_data_frame()
                #Subset to get the good points
                goodpts = localmatches[localmatches['ambiguity'] <= ratio]
                # Get the mean of the good points and then get the point closest to the mean
                meanx, meany = goodpts[['xpos', 'ypos']].mean()
                mid = np.array([[meanx, meany]])
                dists = cdist(mid, goodpts[['xpos', 'ypos']])
                closest = goodpts.iloc[np.argmin(dists)]
                soriginx, soriginy = closest[['xpos', 'ypos']]
                doriginx, doriginy = closest[['match_xpos', 'match_ypos']]

                # Check that we are not within 3 pixels of the edge
                if mindy + buf_dist <= doriginy <= maxdy - buf_dist\
                 and mindx + buf_dist <= doriginx <= maxdx - buf_dist:
                    # Point is good to split on
                    decompose = True
                    break
                else:
                    scounter += 1
                    if scounter >= maxiterations:
                        break
            # We have no guarantee that the match is good, so now check that
            # the match is within the subimage
            if decompose:
                DecomposeAndMatch(cimg1, cimg2,
                                  cmem1, cmem2,
                                  soriginx, soriginy,
                                  doriginx, doriginy,
                                  start,
                                  source_extent, destination_extent)
                # Remove these readbacks if not returning!
                #cmem1.Readback()
                #cmem2.Readback()
                start += 4
    cmem1.Readback()
    cmem2.Readback()
    return mem1, mem2
    #with nogil:
        #DecomposeAndMatch(cimg1, cimg2)

def PyInitCuda(device_number=0):
    """
    Initialize a CUDA GPU

    Parameters
    ----------
    device_number : int
                    The id of the device to initialize.  If the id is greater
                    than the id request, the GPU with the largest id is
                    initialized.
    """
    InitCuda(device_number)

def checkError(error, msg):
    if error != 0:
        raise RuntimeError("Cuda error %d: %s" % (error, msg))

cdef class PySiftData:
    """
    A wrapper around CudaSift's SiftData object
    """
    cdef:
        SiftData data

    def __init__(self, int num = 1024):
        with nogil:
            InitSiftData(self.data, num, False, True)

    def __deallocate__(self):
        with nogil:
            FreeSiftData(self.data)

    def __len__(self):
        return self.data.numPts

    def to_data_frame(self):
        '''Convert the device-side SIFT data to a Pandas data frame and array

        returns a Pandas data frame with the per-keypoint fields: xpos, ypos,
            scale, sharpness, edgeness, orientation, score and ambiguity
            AND a numpy N x 128 array of the SIFT features per keypoint
        '''
        cdef:
            SiftData *data = &self.data
            SiftPoint *pts
            void *dest
            size_t data_size = data.numPts * sizeof(SiftPoint)
            int error
            int state
            np.ndarray[np.float32_t, ndim=2, mode='c'] h_data
            np.ndarray[np.int32_t, ndim=1, mode='c'] match_data
            size_t idx
        nKeypoints = data.numPts;
        if nKeypoints == 0:
            empty = np.zeros(0, np.float32)
            return pandas.DataFrame(dict(
                xpos=empty, ypos=empty, scale=empty,
                sharpness=empty, edgeness=empty, orientation=empty,
                score=empty, ambiguity=empty, match=np.zeros(0, int),
                match_xpos=empty, match_ypos=empty, match_error=empty,
                subsampling=empty)), np.zeros((0, 128), np.float32)
        stride = sizeof(SiftPoint) / sizeof(float)
        dtype = np.dtype("f%d" % sizeof(float))
        h_data = np.ascontiguousarray(np.zeros((nKeypoints, stride), dtype))
        match_data = np.ascontiguousarray(np.zeros(nKeypoints, np.int32))
        pts = <SiftPoint *>h_data.data
        assert h_data.size * sizeof(float) == data_size, ("h_data.size = %d, data_size = %d" % (h_data.size, data_size))
        with nogil:
            state = 0
            error = cudaHostRegister(<void *>pts, data_size, 0)
            if error == 0:
                state = 1
                error = cudaMemcpy(pts, data.d_data, data_size,
                                   cudaMemcpyDeviceToHost)
                cudaHostUnregister(pts)
        checkError(error, "during " + ("cudaHostRegister" if state == 0 else "cudaMemcpy"))
        for 0 <= idx < nKeypoints:
            match_data[idx] = pts[idx].match
        xpos_off = <size_t>(&pts.xpos - <float *>pts)
        ypos_off = <size_t>(&pts.ypos - <float *>pts)
        scale_off = <size_t>(&pts.scale - <float *>pts)
        sharpness_off = <size_t>(&pts.sharpness - <float *>pts)
        edgeness_off = <size_t>(&pts.edgeness - <float *>pts)
        orientation_off = <size_t>(&pts.orientation - <float *>pts)
        score_off = <size_t>(&pts.score - <float *>pts)
        ambiguity_off = <size_t>(&pts.ambiguity - <float *>pts)
        match_xpos_off = <size_t>(&pts.match_xpos - <float *>pts)
        match_ypos_off = <size_t>(&pts.match_ypos - <float *>pts)
        match_error_off = <size_t>(&pts.match_error - <float *>pts)
        subsampling_off = <size_t>(&pts.subsampling - <float *>pts)
        return pandas.concat((
            pandas.Series(h_data[:, xpos_off], name="xpos"),
            pandas.Series(h_data[:, ypos_off], name="ypos"),
            pandas.Series(h_data[:, scale_off], name="scale"),
            pandas.Series(h_data[:, sharpness_off], name="sharpness"),
            pandas.Series(h_data[:, edgeness_off], name="edgeness"),
            pandas.Series(h_data[:, orientation_off], name="orientation"),
            pandas.Series(h_data[:, score_off], name="score"),
            pandas.Series(h_data[:, ambiguity_off], name="ambiguity"),
            pandas.Series(match_data, name = "match"),
            pandas.Series(h_data[:, match_xpos_off], name="match_xpos"),
            pandas.Series(h_data[:, match_ypos_off], name="match_ypos"),
            pandas.Series(h_data[:, match_error_off], name="match_error"),
            pandas.Series(h_data[:, subsampling_off], name="subsampling")
            ), axis=1), h_data[:, -128:]

    @staticmethod
    def from_data_frame(data_frame, features):
        """
        Set a SiftData from a data frame and feature vector

        data_frame : DataFrame
                     a Pandas data frame with the per-keypoint fields:
                     xpos, ypos, scale, sharpness, edgeness, orientation, score
                     and ambiguity
        features : ndarray
                   (n,128) array of SIFT features
        """
        assert len(data_frame) == len(features)
        self = PySiftData(len(data_frame))
        cdef:
            SiftData *data = &self.data
            SiftPoint *pts
            size_t size = len(data_frame)
            int state
            int error
            size_t data_size
            np.ndarray[np.float32_t, ndim=2, mode='c'] tmp

        tmp = np.ascontiguousarray(np.zeros(
            (size, sizeof(SiftPoint) / sizeof(float)),
            dtype="f%d" % sizeof(float)))
        pts = <SiftPoint *>tmp.data
        xpos_off = <size_t>(&pts.xpos - <float *>pts)
        ypos_off = <size_t>(&pts.ypos - <float *>pts)
        scale_off = <size_t>(&pts.scale - <float *>pts)
        sharpness_off = <size_t>(&pts.sharpness - <float *>pts)
        edgeness_off = <size_t>(&pts.edgeness - <float *>pts)
        orientation_off = <size_t>(&pts.orientation - <float *>pts)
        score_off = <size_t>(&pts.score - <float *>pts)
        ambiguity_off = <size_t>(&pts.ambiguity - <float *>pts)
        tmp[:, xpos_off] = data_frame.xpos.as_matrix()
        tmp[:, ypos_off] = data_frame.ypos.as_matrix()
        tmp[:, scale_off] = data_frame.scale.as_matrix()
        tmp[:, sharpness_off] = data_frame.sharpness.as_matrix()
        tmp[:, edgeness_off] = data_frame.edgeness.as_matrix()
        tmp[:, orientation_off] = data_frame.orientation.as_matrix()
        tmp[:, score_off] = data_frame.score.as_matrix()
        tmp[:, ambiguity_off] = data_frame.ambiguity.as_matrix()
        tmp[:, -128:] = features
        data.numPts = size
        data_size = size * sizeof(SiftPoint)
        with nogil:
            error = cudaMemcpy(data.d_data, pts, data_size,
                                   cudaMemcpyHostToDevice)
        checkError(error, "during " + ("cudaHostRegister" if state == 0 else "cudaMemcpy"))
        return self

def ExtractKeypoints(np.ndarray srcImage,
                     PySiftData pySiftData,
                     int numOctaves = 5,
                     float initBlur = 0,
                     float thresh = 5,
                     float lowestScale = 0,
                     float subsampling = 1.0):
    """
    Extract keypoints from an image

    Parameters
    ----------
    srcImage : ndarray
               (n, m) image array

    pySiftData : object
                 PySiftData object in which to store pts

    numOctaves : int
                 # of octaves to accumulate

    initBlur : int
               the initial Gaussian standard deviation

    thresh : int
             significance threshold for keypoints

    lowestScale: int
                 The ?

    subsampling : float
                  subsampling in pixels

    Returns
    -------
    returns a pandas data frame of SIFT points and an N x 128 numpy array of
        SIFT features per keypoint
    """
    cdef:
        size_t i
        SiftPoint *pts
        CudaImage destImage
        size_t lim = srcImage.size
        size_t size_x = srcImage.shape[1]
        size_t size_y = srcImage.shape[0]
        np.ndarray tmp = np.ascontiguousarray(srcImage.astype(np.float32))
        void *pSrc = tmp.data
    with nogil:
        destImage.Allocate(size_x, size_y, iAlignUp(size_x, 128),
                         False, NULL, <float *>pSrc)
        destImage.Download()
    del tmp
    with nogil:
        ExtractSift(pySiftData.data, destImage, numOctaves, initBlur, thresh,
                    lowestScale, subsampling)

cdef class PyCudaImage(object):
    """
    Wrapper around CudaSift's CudaImage object
    """

    def __init__(self, np.ndarray img):
        cdef:
            CudaImage destImage
            size_t size_x = img.shape[1]
            size_t size_y = img.shape[0]
            np.ndarray tmp = np.ascontiguousarray(img.astype(np.float32))
            void *pSrc = tmp.data
        with nogil:
            destImage.Allocate(size_x, size_y, iAlignUp(size_x, 128),
                               False, NULL, <float *>pSrc)
            destImage.Download()
        print('Downloaded')
        del tmp

def PyMatchSiftData(PySiftData data1, PySiftData data2):
    """
    Given two PySiftData objects, apply the CUDA matcher.

    Parameters
    ----------
    data1 : object
            PySiftData object

    data2 : object
            PySiftData object
    """
    with nogil:
        MatchSiftData(data1.data, data2.data)
