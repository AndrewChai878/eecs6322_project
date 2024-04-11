# EECS6322 Reproducibility Project

By: Andrew Chai & Alexander Bianchi

Paper: [CoordX: Accelerating Implicit Neural Representation with a Split MLP Architecture](https://openreview.net/forum?id=oAy7yPmdNz) (ICLR 2022)

### Set up:
We used Python 3.11 with torch 2.2 cuda 12.1

You will want to have the following additional packages:

`pip install av` (for video experiments)

`pip install siren-pytorch` (for sine activation function)

### Project Info

To run the code you only need to access `coordx.py` for the reproduced model and `helpers.py` for the training function with accelerated sampling.

The `example.ipynb` notebook contains a walkthrough of how to train CoordX to represent images and videos. The other notebooks in this repository are there to show our work over time.

The original paper uses 12 random images from the DIV2K dataset to evaluate image representation. We inlcude the 12 DIV2K images we randomly sampled and used for our evaluation in the images folder. Images used in our report and analyis are available in the output_images folder. 

The two 10-second video clips that the original authors used to evaluate CoordX for video are located in the videos folder. The output of the CoordX models representing these videos that we refer to in our report are located in the output_videos folder.