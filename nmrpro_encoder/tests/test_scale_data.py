import numpy.testing as ts
import unittest
from nmrpro.readers import fromFile
from base64 import b64encode
from nmrpro_encoder import encoder
from nmrpro_encoder import decoder
import numpy as np
from nose_parameterized import parameterized
import itertools

class scaleDataTest(unittest.TestCase):
    def setUp(self):
        self.spec1d = fromFile("/Users/mohamedahmed/NMR/data/Bruker/expnmr_00001_1.tar").real

    @parameterized.expand(itertools.product(
        # arr having a negative value, or all postives
        [[ 256,  512, 1025,   255, -257], [ 256,  512, 1025,   255, 257]],
        [16,8]
    ))
    def test_scaled_postive_negative(self,arr, bits):
        a = np.asarray(arr, dtype=np.int16)
        b = encoder.scale_data(a, bits, "positive")
        re_a = decoder.descale(b, bits, (a.min(), a.max()))

        resolution = a.ptp() / (2.**bits-1)
        ts.assert_allclose( a, re_a, rtol=1, atol=resolution,
            err_msg='Rescaled sample array diviated from the original by more than accepted resolution')

    @parameterized.expand([
        (8,), (16,)
    ])
    def test_scaled_spec(self, bits):
        a = self.spec1d
        b = encoder.scale_data(a, bits, "positive")
        re_a = decoder.descale(b, bits, (a.min(), a.max()))

        resolution = a.ptp() / (2.**bits-1)
        ts.assert_allclose( a, re_a, atol=resolution,
            err_msg='Rescaled spectrum diviated from the original by more than accepted resolution')

class PNGConversionTest(unittest.TestCase):
    def setUp(self):
        self.spec1d = fromFile("/Users/mohamedahmed/NMR/data/Bruker/expnmr_00001_1.tar").real
  
    def test_png8(self):
        a = np.asarray([ 0,  127, 200,   255, 20], dtype=np.int16)
        a = np.uint8(a)
        a.shape = (a.shape[0], 1)
        b = encoder.get_data_as_png(a)
        b = b64encode(b.getvalue())
        re_a = decoder.decode_png(b)
        re_a.shape = a.shape
        
        ts.assert_array_equal(a, re_a, 
                    'Sample array reconstructed do not match the original')
    
    def test_png16(self):
        a = np.asarray([ 0,  127, 200,   255, 20], dtype=np.int16)
        a = encoder.scale_data(a, 16, "positive")
        a = np.uint16(a)
        
        b = a.view(np.uint8)
        b = np.concatenate( (b[0::2],b[1::2]) )
        b.shape = (b.shape[0], 1)

        c = encoder.get_data_as_png(b)
        c = b64encode(c.getvalue())
        re_a = decoder.decode_png16(c)
        re_a.shape = a.shape
        
        ts.assert_array_equal(a, re_a, 
                    'Sample array reconstructed do not match the original')

    
    def test_png8_spec(self):
        a = self.spec1d
        a = encoder.scale_data(a, 8, "positive")
        a = np.uint8(a)
        a.shape = (a.shape[0], 1)
        b = encoder.get_data_as_png(a)
        b = b64encode(b.getvalue())
        re_a = decoder.decode_png(b)
        re_a.shape = a.shape
        ts.assert_array_equal(a, re_a, 
                        'Spectrum reconstructed do not match the original')
        
    def test_png8_spec_reshape(self):
        a = self.spec1d
        a = encoder.reshape_by_factor(a)
        a = encoder.scale_data(a, 8, "positive")
        a = np.uint8(a)
        b = encoder.get_data_as_png(a)
        b = b64encode(b.getvalue())
        re_a = decoder.decode_png(b)
        
        # reset "a" shape
        a = a.flatten()
        ts.assert_array_equal(a, re_a, 
                        'Spectrum reconstructed do not match the original')

    def test_png16_spec(self):
        a = self.spec1d
        a = encoder.scale_data(a, 16, "positive")
        a = np.uint16(a)
        
        b = a.view(np.uint8)
        b = np.concatenate( (b[0::2],b[1::2]) )
        b.shape = (b.shape[0], 1)
        
        
        c = encoder.get_data_as_png(b)
        c = b64encode(c.getvalue())
        re_a = decoder.decode_png16(c)
        
        # reset a
        a = a.flatten()
        re_a.shape = a.shape
        ts.assert_array_equal(a, re_a, 
                        'Spectrum reconstructed do not match the original')


class EncodeSpecTest(unittest.TestCase):
    def setUp(self):
        self.spec1d = fromFile("/Users/mohamedahmed/NMR/data/Bruker/expnmr_00001_1.tar").real

    @parameterized.expand([
        (8,), (16,)
    ])
    def test_encode(self, bits):
        format = "png" if bits == 8 else "png16"
        
        a = self.spec1d
        b = encoder.encode1DArrayAsPNG(a, format)
        
        re_a = decoder.decode_png_array(b, format, (a.min(), a.max()) )
        
        resolution = a.ptp() / (2.**bits-1)
        
        ts.assert_allclose( a, re_a, atol=resolution,
            err_msg='Rescaled spectrum diviated from the original by more than accepted resolution')

