from setuptools import find_packages, setup
setup(
    name='ASLInterpreter',
    packages=find_packages(),
    version='0.1.0',
    description='A framework to predict American Sign Language',
    author='ASL Inter',
    license='MIT',
    install_requires=['mediapipe'],
    zip_safe=False
)