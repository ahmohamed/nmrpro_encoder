from json import dumps, loads
import encoder
from nmrpro.classes import NMRSpectrum
from nmrpro.decorators import perSpectrum

@perSpectrum
def to_json(spec, format="png16"):
    'Returns a PNG encoded spectrum as a JSON string.'
    
    encoding = {
        "png": encoder.pngSpecEncoder, 
        "png16": encoder.png16SpecEncoder
    }[format]
    return dumps(ret, cls=encoding)

@perSpectrum
def to_dict(spec, format="png16"):
    'Returns a PNG encoded spectrum as a python dict object.'
    
    return encoder.encodeSpec(spec, format)
    
def from_json(json_spec):
    '''Converts a PNG encoded spectrum (JSON string) to a numpy array. 
    An NMRSpectrum cannot be constructed becuase some udic information is lost.'''
    
    # TODO: check data_type == spectrum, format == png
    parsed_json = loads(json_spec)
    
    data = encoder.decode_png_array(parsed_json['data'], parsed_json['bits'], parsed_json['y_domain'])
    return data
    
