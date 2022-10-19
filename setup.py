from setuptools import setup, find_packages

setup(
    name='controllable-video-gen',
    version='1.0.0',
    description='Layered Controllable Video Generation',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
