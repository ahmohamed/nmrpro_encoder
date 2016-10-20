from PIL import Image
from io import BytesIO
import numpy as np

def decode_png(b64str):
    # Decode the base64 png and get pixel data
    decoded_png = b64str.decode('base64')
    img_data = BytesIO()
    img_data.write(decoded_png)
    img_data.flush()
    img_data.seek(0)
    img = Image.open(img_data)
    img_arr = np.asarray(list(img.getdata()), dtype=np.uint8)
    
    return img_arr


def decode_png16(b64str):
    img_arr = decode_png(b64str)
    out = np.empty_like(img_arr)
    out[0::2] = img_arr[0:out.shape[0]/2]
    out[1::2] = img_arr[out.shape[0]/2:out.shape[0]]
    out = out.view(np.uint16)
    return out


def descale(obj, bits, domain):
    _range = domain[1] - domain[0]
    resolution = _range / (2.**bits-1)
    scaled = (obj * resolution) + domain[0]
    return scaled


def decode_png_array(b64str, format, domain):
    decoder = {
        "png":decode_png,
        "png16":decode_png16
    }[format]
    bits = {
        "png":8,
        "png16":16
    }[format]
    
    img_arr = decoder(b64str)
    out = descale(img_arr, bits,domain)
    return out
