from .utils import *

def getBokeh(img_path, layers, focalLength=1.0, DoF=0.3, fStop=2.0):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = toOne(img)

    output = np.zeros_like(img)

    for layer in layers:
        if layer < focalLength - min(focalLength, DoF) or layer > focalLength:
            d = abs(focalLength - layer) / fStop
            maxD = 0.83
            maxSize = 100
            size = (d * maxSize) / maxD
            blurred = lensBlur(img, radius=size/2)
            imgMask = layers[layer] * blurred
        else:
            imgMask = layers[layer] * img
    output = output + imgMask

    return output