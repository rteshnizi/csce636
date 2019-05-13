# Image Synthesis Based on Semantic Layout

## Python Requirements

The following Python libraries are needed to run the codes in this repository:
* Tensorflow (GPU)
* NumPy
* SciPy
* Pillow
* PyTz

## Large Files

There are two directories that are empty in this repository: [`demo`](demo/) and [`VGG_Model`](VGG_Model/).

For the codes to work the files inside these directories need to be downloaded separately from the provided URL in the README files in the respective directories.

## Data set

The data set for training this network is not uploaded as part of this repository due the large volume and size of the data set.
The data set is available for download at https://www.cityscapes-dataset.com/downloads/.
For training, the semantic layouts in `gtFine_trainvaltest.zip` (241MB) and the RGB images in `leftImg8bit_trainvaltest.zip` (11GB) are required.

## Training the Network

To train the network, the label and training files should be placed in their corresponding directories, and then run `python train.py`

## Running GUI

To run GUI, you need a PC with a GPU with at least 2GB of free memory.

A video demoing the GUI is available at https://youtu.be/nLnGzxoUMj0

1. Run `python gui.py`
2. Wait for the model to be loaded. This might take up to a minute depending on the PC's performance.
3. Click on Select layout.
4. In the file browser menu got to `input` folder and select one of the sample test data images.
5. Click RUN! to run the image through the trained network.
6. The first run would take up to a minute depending on the PC's performance.
7. The synthesized image will appear in the right hand side of the GUI panel.

## Validation Data

The below animation illustrates the quality of the output of one image in the validation set over the span of 200 epochs.

https://youtu.be/Tmp1ZuYSj88



