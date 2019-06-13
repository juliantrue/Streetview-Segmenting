# Streetview Segmenting

This is a project in progress to segment out objects from Google Streetview data. It is built on top of the Matterport Mask RCNN implemetation found her
[here](https://github.com/matterport/Mask_RCNN).

# Usage
## Running in Docker
For complete portability, use the included Dockerfile. The only requirement is that it is run on a system with an Nvidia GPU and a working installation of [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

Build the docker image `make build-image` and run the container with `make run-image`.
If you wish to attach multiple bash instances to the running Docker container, simply run `make run-image` again to attach.

## Training
In order to train on the provided building facade dataset, run the `make train_mask_RCNN`! Simple 


**Thats it!**
