from distutils.core import setup
from setuptools import find_packages
from setuptools.command.install import install
import os

setup(
    name='leap_layer',
    version='1.0.0',
    description='Leap Hand kinematics layer',
    author='Wei Wei',
    author_email='wei.wei@cair-cas.org.hk',
    url='wei.wei2018@ia.ac.cn',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch1.10.0',
        'trimesh',
        'roma',
        'pytorch-kinematics',
        'mesh-to-sdf',
        'point-cloud-utils'
    ]
)

