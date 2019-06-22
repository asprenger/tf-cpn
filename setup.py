
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='tfcpn',
    version='0.0.1',
    description='Tensorflow implementation of CPN (Cascaded Pyramid Network)',
    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'data', 'lib', 'models', 'notebooks']),
    install_requires=[
        'tensorflow-gpu == 1.13.1;platform_system=="Linux"',
        'tensorflow == 1.13.1;platform_system=="Darwin"'
    ]
)    
