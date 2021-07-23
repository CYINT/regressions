# _*_ coding: utf-8 _*_
from setuptools import find_packages
from distutils.core import setup

setup(
    name='cyint-regressions',
    version='1.0.0',
    author='Daniel Fredriksen',
    author_email='dfredriksen@cyint.technology',
    packages=find_packages(),
    url='https://github.com/CYINT/regressions',
    license='MIT',
    description='Helper module to automatically create the best regressions for your data'
)