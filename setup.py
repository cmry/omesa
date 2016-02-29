"""Setup file."""

from setuptools import setup
from setuptools import find_packages


setup(name='omesa',
      version='0.2.8.alpha',
      description='A small framework for reproducible text mining research',
      author='Chris Emmery',
      author_email='chris.emmery@uantwerpen.be',
      url='https://github.com/cmry/shed',
      license='MIT',
      packages=find_packages(exclude=['tests']),
      install_requires=['scikit-learn>=0.17'],
      zip_safe=True)
