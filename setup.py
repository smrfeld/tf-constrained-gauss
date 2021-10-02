import setuptools
from distutils.core import setup

with open("README_pypi.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='tfConstrainedGauss',
    version='0.1.0',
    packages=['tfConstrainedGauss','tfConstrainedGauss/solve_id','tfConstrainedGauss/solve_me'],
    author='Oliver K. Ernst',
    author_email='oliver.k.ernst@gmail.com',
    url='https://github.com/smrfeld/tf-constrained-gauss',
    python_requires='>=3.7',
    description="TensorFlow package for estimating constrained precision matrices",
    long_description=long_description,
    long_description_content_type="text/markdown"
    )