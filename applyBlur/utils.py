import numpy as np
import cv2

def toOne(img):
  return img.astype('float') / 255

def to255(img): 
  return (img * 255).astype('uint8')
  
def applyBlur(img, size=(256, 256), radius=50):
  sizeX, sizeY = size
  x,y = np.mgrid[-sizeY/2:sizeY/2, -sizeX/2:sizeX/2]
  disk = (np.sqrt(x**2 + y**2) < radius).astype(float)
  disk /= disk.sum()
  blurred = cv2.filter2D(img, -1, disk)
  return blurred

def applyBloom(img, threshold, gain):
  gray = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2GRAY)
  mask = gray >= threshold
  img[mask] *= gain
  return img

def lensBlur(img, radius=50, gamma=3.0):
  size = radius * 4
  gammaCorrectedImg = np.power(img, gamma)
  bokehImg = applyBlur(gammaCorrectedImg, (size, size), radius)
  bokehImg = np.cbrt(bokehImg)
  blurImg = applyBlur(img, (size, size), radius)
  finalImg = np.maximum(blurImg, bokehImg)
  return finalImg
