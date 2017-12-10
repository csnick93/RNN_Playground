# RNN_Playground

This is a small keras based RNN playground for text understanding and generation. The code is written in Python 3. It is meant to easily play around with different RNN architectures (SimpleRNN, GRU, LSTM) to understand the RNN's powerful text understanding and generation capacity. The results of the trained RNN models can be visualized nicely in a html file, using a simple GUI for generation.
The models in the result folder currently are based on training on a long Sherlock Holmes text file (The Project Gutenberg).

## Getting Started

Download the repository. 
For a quick start:
  - Adjust the config.xml file to the configuration of the RNN architecture you prefer
  - Run the RNN code using the main.py
  - Run the htmlReportGUI.py to get a HTML overview of the trained models
  
## Prerequisites

The following packages are required in order to run the code:
  - keras (>= 2.0.6)
  - numpy (>= 1.13.3)
  
 The following required packages you should have per default:
  - tkinter 
  - matplotlib
  - xml
  - webbrowser
  
## Configuring the RNN model

Open the config.xml in the Configuration folder. There, the learning parameters, the sampling parameters (for text generation) and Paths (the textfile to learn from) can be configured.

## Starting the training
Adjust the config.xml to your desires and run main.py. A new folder in Results will be created and the results will be saved there. The results consist of:
  - history.txt: The loss over the epochs
  - loss_plot.png: The loss visualized
  - model.h5: The best model saved (based on validation loss)
  - sampleText.txt: The generated text after training
  
## Visualizing results
In order to get a nice summary of the trained models, run htmlReportGUI.py. A GUI will pop up listing all the different configurations encountered in the Results folder. Choose the configurations you are interested in. After a html file will automatically open in your browser so you can view the results.
  
