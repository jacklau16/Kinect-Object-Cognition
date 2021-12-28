# CSMVI16 Coursework 2

This report is written for the coursework 2 of the module CSMVI16 – Visual Intelligence. The objective of the coursework is to implement an object recognition method in Python to recognise different objects. The appearance-based method is adopted in the implementation.

In developing the object recognition system, the following tools/libraries are used:
### `Python`: Programming language for developing the system
### `Fakenect`: Library to read the video data dumped from a real Kinect
#### `OpenCV`: Real-time computer vision library, used for image processing in the object recognition system

The development is done under a virtual environment of `PyCharm`, the package dependency is maintained by `poetry`, and those dependancies are saved in the file `pyproject.toml`.

Python version: 3.8

## Prepare training images
Run the following command:

`python capture_train_img.py [Set 1 video folder]`

The captured training images will be saved in the folder `./train_img/`.
## Train and test the object recognition
Run the following command:

`python run_obj_recognition.py [Set 2 video folder]`
