import numpy as np
import cv2

def toOne(img):
  return img.astype('float') / 255

def to255(img): 
  return (img * 255).astype('uint8')

def getDisk(x, y, radius):
  disk = (np.sqrt(x**2 + y**2) < radius).astype(float)
  disk /= disk.sum()
  return disk

def getKite(x, y, l):
  kite = (np.abs(x) + np.abs(y) < l).astype(float)
  kite /= kite.sum()
  return kite

def getSquare(x, y, l):
  square = (np.maximum(np.abs(x), np.abs(y)) < l).astype(float)
  square /= square.sum()
  return square

def getTriangle(x, y, l):
  triangle = (np.maximum(np.abs(4*x) + 2*y, np.abs(-2*y)) <= l).astype(float)
  triangle /= triangle.sum()
  return triangle

def getTrapezoid(x, y, l):
  trapezoid = (np.maximum(np.abs(5*y), np.abs(4*x) + 2*y) <= l*2).astype(float)
  trapezoid /= trapezoid.sum()
  return trapezoid

shapeDict = {
  "disk": getDisk,
  "kite": getKite,
  "square": getSquare,
  "triangle": getTriangle,
  "trapezoid": getTrapezoid,
}
  
def applyBlur(img, size=(256, 256), radius=50, shape="disk"):
  sizeX, sizeY = size
  x,y = np.mgrid[-sizeY/2:sizeY/2, -sizeX/2:sizeX/2]
  filter = shapeDict[shape](x, y, radius)
  blurred = cv2.filter2D(img, -1, filter)
  return blurred

def lensBlur(img, radius=50, shape="disk", gamma=3.0):
  size = radius * 4
  gammaCorrectedImg = np.power(img, gamma)
  bokehImg = applyBlur(gammaCorrectedImg, (size, size), radius, shape)
  bokehImg = np.cbrt(bokehImg)
  blurImg = applyBlur(img, (size, size), radius, shape)
  finalImg = np.maximum(blurImg, bokehImg)
  return finalImg
