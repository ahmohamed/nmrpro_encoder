from distutils.core import setup
from os.path import exists
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(
  name = 'nmrpro_encoder',
  packages = find_packages(), # this must be the same as the name above
  include_package_data=True,
  platforms='any',
  version = '0.0.1',
  description = 'Companion package for encoding and deconding nmrpro spectra.',
  author = 'Ahmed Mohamed',
  author_email = 'mohamed@kuicr.kyoto-u.ac.jp',
  install_requires=['nmrpro', 'Pillow'],
  url = 'https://github.com/ahmohamed/nmrpro_encoder',
  license='MIT',
  keywords = ['nmr', 'spectra', 'multi-dimensional', 'png-format'],
  classifiers = [],
)