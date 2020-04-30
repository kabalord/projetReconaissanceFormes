#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 21:33:35 2020

@author: walterroaserrano
"""

## Call the opencv library and numpy
import cv2
import numpy as np

## RGB image
imgColor = cv2.imread('/Users/walterroaserrano/Desktop/UniversiteChampagneArdenne/reconnaissanceFormes/projetReconaissanceFormes/example1.jpg')

## Grey image 
img = cv2.imread('/Users/walterroaserrano/Desktop/UniversiteChampagneArdenne/reconnaissanceFormes/projetReconaissanceFormes/example1.jpg', 0)

## To show the image
cv2.imshow('Colored Image', imgColor)
cv2.imshow('Grey Image', img)
cv2.waitKey (0)
cv2.destroyAllwindows()

## The image size
print (imgColor.shape)
print (img.shape)

## The number of pixels 
img.size

## Write an image 
cv2.imwrite('gray image.png', img)
