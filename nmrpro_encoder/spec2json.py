from json import dumps, loads
import encoder
from nmrpro.classes import NMRSpectrum

def to_json(spec, format="png16"):
    encoding = {
        "png": encoder.pngSpecEncoder, 
        "png16": encoder.png16SpecEncoder
    }[format]
    return dumps(ret, cls=encoding)

def from_json(json_spec):
    # check data_type == spectrum, format == png
    parsed_json = loads(json_spec)
    
    data = encoder.decode_png_array(parsed_json['data'], parsed_json['bits'], parsed_json['y_domain'])
    return data
    
