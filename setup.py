from setuptools import setup
from setuptools import find_packages

from tensortools.function import Function, function
from tensortools.model import Model, model

setup(name='chiRLalg',
        version='0.3',
        description='',
        author='Simon Ramstedt, Juliette Tibayrenc'
        author_email='simonramstedt@gmail.com',
        url='https://github.com/jtib/chi-rl-alg',
        download_url='',
        license='MIT',
        install_requires=['tensorflow>=1.0.1', 'matplotlib>=2.0', 'flask>=0.12', 'flask_socketio', 'watchdog', 'gym'],
        packages=find_packages())


