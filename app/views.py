from xmlrpc.client import Boolean
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.http import HttpResponse
from django.conf import settings

from rest_framework.views import APIView
from applyBlur import getBokeh, utils

import os
import ast
import cv2
import base64
import math
from PIL import Image
import numpy as np

# Create your views here.
class BlurView(APIView):
    """
    Make prediction
    """
    def __init__(self):
        self.focal_lengths = [math.floor((x * 0.1) * 1e12) / 1e12 for x in range(0, 11)]
        self.layers = {}

    def getImagePath(self, image, name):
        try:
            path = default_storage.save(f'images/{name}.jpg', ContentFile(image.read()))
        except:
            format, imgstr = image.split(';base64,')
            ext = format.split('/')[-1] 
            image = base64.b64decode(imgstr)
            path = default_storage.save(f'images/{name}.jpg', ContentFile(image))
        return path

    def post(self, request):
        data = dict(request.data)
        rgb_image = data['rgb_image'][0]
        depth_image = data['depth_image'][0]
        focal_length = float(data['focal_length'][0])
        dof = float(data['dof'][0])
        f_stop = float(data['f_stop'][0])
        shape = data['shape'][0]
        is_image_new = Boolean(data['is_image_new'][0])

        rgb_path = self.getImagePath(rgb_image, "rgb")
        depth_path = self.getImagePath(depth_image, "depth")
        rgb_path = os.path.join(settings.MEDIA_ROOT, rgb_path)
        depth_path = os.path.join(settings.MEDIA_ROOT, depth_path)

        if is_image_new:
            depth = cv2.imread(depth_path)
            depth = cv2.cvtColor(depth, cv2.COLOR_RGB2BGR)
            depth = utils.toOne(depth)
            depth = cv2.GaussianBlur(depth,(51,51),0)
            for i in range(len(self.focal_lengths)-1):
                mask = ((depth >= self.focal_lengths[i]) & (depth < self.focal_lengths[i+1])).astype("float")
                self.layers[self.focal_lengths[i]] = (mask > 0.5).astype("float")

        output = getBokeh.getBokeh(rgb_path, self.layers, focal_length, dof, f_stop, shape)
        response = HttpResponse(content_type='image/jpg')
        output = Image.fromarray(np.uint8(output*255))
        output.save(response, "JPEG")
        os.remove(rgb_path)
        os.remove(depth_path)
        return response