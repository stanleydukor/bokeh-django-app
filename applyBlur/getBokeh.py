from .utils import *

def getBokeh(img_path, layers, focal_length=1.0, dof=0.3, f_stop=2.0):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = toOne(img)

    output = np.zeros_like(img)

    for layer in layers:
        if layer < focal_length - min(focal_length, dof) or layer > focal_length:
            d = abs(focal_length - layer) / f_stop
            maxD = 0.83
            maxSize = 100
            size = (d * maxSize) / maxD
            blurred = lensBlur(img, radius=size/2)
            imgMask = layers[layer] * blurred
        else:
            imgMask = layers[layer] * img
    output = output + imgMask

    return output