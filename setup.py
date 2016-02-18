"""Setup file."""

from setuptools import setup
from setuptools import find_packages


setup(name='shed',
      version='0.0.1',
      description='A small framework for reproducible text mining research',
      author='Chris Emmery',
      author_email='chris.emmery@uantwerpen.be',
      url='https://github.com/cmry/shed',
      download_url='https://github.com/cmry/shed/tarball/0.0.1',
      license='MIT',
      install_requires=['scikit-learn'],
      packages=find_packages())
