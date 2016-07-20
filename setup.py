"""Setup file."""

from setuptools import setup
from setuptools import find_packages


setup(name='omesa',
      version='0.3.1.alpha',
      description='Framework for reproducible machine learning research',
      author='Chris Emmery',
      author_email='chris.emmery@uantwerpen.be',
      url='https://github.com/cmry/omesa',
      license='MIT',
      packages=find_packages(exclude=['examples']),
      install_requires=['blitzdb>=0.2.12',
                        'bottle>=0.12.9',
                        'lime>=0.1.1.4',
                        'plotly>=1.9.9',
                        'colorlover>=0.2.1',
                        'scikit-learn>=0.17'],
      zip_safe=True)
