  This project uses the pytorch GATv2 framework to create a go engine. 
  The training data is from katagotraining.org, trained from the latest katago self play games.
  All the training up to now is run on a 5080 using cuda 13.2, python 14.3, and pytorch 2.10.0 + cu128
  The python folder contains the training pipeline, graph creation and model architecture, the data_manipulation folder contains python scripts
that loads training files and convert them into graph data contained in .pt files. the interactive folder enables connection to online-go.com
