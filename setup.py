"""Setup file."""

from setuptools import setup
from setuptools import find_packages


setup(name='omesa',
      version='0.3.2.alpha',
      description='A framework for reproducible machine learning research',
      author='Chris Emmery',
      author_email='chris.emmery@uantwerpen.be',
      url='https://github.com/cmry/omesa',
      license='GPLv3',
      packages=find_packages(exclude=['tests']),
      install_requires=['blitzdb>=0.2.12',
                        'bottle>=0.12.9',
                        'colorlover>=0.2.1',
                        'lime>=0.1.1.7',
                        'plotly>=1.12.0',
                        'reach>=0.0.4',
                        'scikit-learn>=0.17'],
      zip_safe=True)
