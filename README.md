# Image Synthesis Based on Semantic Layout

## Training the Network

The dataset for training this network is not uploaded as part of this repository due the large volume and size of the dataset. I can provide the training, validation, and test data upon request. Additionally, the size of the checkpoint file (Tensorflow session) is 1.2 GB. Github has a limit of 100MB size limit per file.

To train the network, run `python train.py`

## Running GUI

To run GUI, you need a PC with a GPU with at least 2GB of free memory.

1. Run `python gui.py`
2. Wait for the model to be loaded. This might take up to a minute depending on the PC's performance.
3. Click on Select layout.
4. In the file browser menu got to `input` folder and select one of the sample test data images.
5. Click RUN! to run the image through the trained network.
6. The first run would take up to a minute depending on the PC's performance.
7. The synthesized image will appear in the right hand side of the GUI panel.
