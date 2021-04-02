from setuptools import find_packages, setup

setup(
    name='ASLInterpreter',
    packages=find_packages(),
    version='0.0.1',
    description='A framework to predict American Sign Language',
    author='FOSSLife Foundation',
    license='MIT',
    install_requires=['mediapipe', 'opencv-python', 'keras', 'tensorflow']
)