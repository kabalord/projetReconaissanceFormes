#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 20:24:33 2020

@author: walterroaserrano
"""

import numpy
import cv2

## Implémentation LDP Kirsh Algorithm 
KIRSCH_K1   = numpy.array([[ 5, -3, -3], [ 5,  0, -3], [ 5, -3, -3]], dtype=numpy.float32) / 15
KIRSCH_K2   = numpy.array([[-3, -3,  5], [-3,  0,  5], [-3, -3,  5]], dtype=numpy.float32) / 15
KIRSCH_K3   = numpy.array([[-3, -3, -3], [ 5,  0, -3], [ 5,  5, -3]], dtype=numpy.float32) / 15
KIRSCH_K4   = numpy.array([[-3,  5,  5], [-3,  0,  5], [-3, -3, -3]], dtype=numpy.float32) / 15
KIRSCH_K5   = numpy.array([[-3, -3, -3], [-3,  0, -3], [ 5,  5,  5]], dtype=numpy.float32) / 15
KIRSCH_K6   = numpy.array([[ 5,  5,  5], [-3,  0, -3], [-3, -3, -3]], dtype=numpy.float32) / 15
KIRSCH_K7   = numpy.array([[-3, -3, -3], [-3,  0,  5], [-3,  5,  5]], dtype=numpy.float32) / 15
KIRSCH_K8   = numpy.array([[ 5,  5, -3], [ 5,  0, -3], [-3, -3, -3]], dtype=numpy.float32) / 15

def kirsch_filter(img) :
    """ Return a gray-scale image that's been Kirsch edge filtered. """
    if  img.ndim > 2 :
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fimg    = numpy.maximum(cv2.filter2D(img, cv2.CV_8U, KIRSCH_K1),
              numpy.maximum(cv2.filter2D(img, cv2.CV_8U, KIRSCH_K2),
              numpy.maximum(cv2.filter2D(img, cv2.CV_8U, KIRSCH_K3),
              numpy.maximum(cv2.filter2D(img, cv2.CV_8U, KIRSCH_K4),
              numpy.maximum(cv2.filter2D(img, cv2.CV_8U, KIRSCH_K5),
              numpy.maximum(cv2.filter2D(img, cv2.CV_8U, KIRSCH_K6),
              numpy.maximum(cv2.filter2D(img, cv2.CV_8U, KIRSCH_K7),
                            cv2.filter2D(img, cv2.CV_8U, KIRSCH_K8),
                           )))))))
    return(fimg)


def threshold(img, sig = None) :
    """ Threshold a gray image in a way that usually makes sense. """
    if  img.ndim > 2 :
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    med     = numpy.median(img)
    if  sig is None :
        sig = 0.0       # note: sig can be negative. Another way: Use the %'th percentile-ish pixel.
    co      = int(min(255, max(0, (1.0 + sig) * med)))
    return(cv2.threshold(img, co, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)[1])



if  __name__ == '__main__' :
    import  os
    import  sys

    fn      = "/Users/walterroaserrano/Desktop/UniversiteChampagneArdenne/reconnaissanceFormes/projetReconaissanceFormes/example1.jpg"
    if  len(sys.argv) > 1 :
        fn  = sys.argv.pop(1)
    sig     = None
    if  len(sys.argv) > 1 :
        sig = float(sys.argv.pop(1))

    img     = cv2.imread(fn)

    kimg    = kirsch_filter(img)    # make each pixel the maximum edginess value
    timg    = threshold(kimg, sig)  # make the edges stand out
    timg    = 255 - timg            # invert the image to make the edges white

    cv2.imshow('%s kirsch filtered' % os.path.basename(fn), timg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

