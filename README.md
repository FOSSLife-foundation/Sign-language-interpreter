# Sign Language Interpreter

## Introduction
The goal of this project is to build a system that acts as a translator for Sign Language, specifically American Sign Language (ASL). 

## How it Works
Frames from a video feed taken from a camera would be given as input to a pair of classification models. The camera would be positioned in front of the signer and as he/she is signing, one of the models would attempt to detect letters of the alphabet while the other would attempt to detect words/expressions in ASL. Both the models would be running on a Raspberry Pi and the video feed would be taken from a Pi camera. The exact architecture in which the two models are used is to be decided based on their individual performance and their composite performance.
