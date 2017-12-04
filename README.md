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
  - keras
  - numpy 
  
 The following required packages you should have per default:
  - tkinter 
  - matplotlib
  - xml
  - os
  - webbrowser
  
## Configuring the RNN model

Open the config.xml in the Configuration folder. There, the learning parameters, the sampling parameters (for text generation) and Paths (the textfile to learn from) can be configured.

### Learning Parameters
  - epochs: Number of epochs to train
  - unit_type: the type of RNN to use (SimpleRNN, GRU, LSTM)
  - no_layers: the number of RNN layers to use
  - seq_length: the unfolding length of the RNN 
  - hidden_size: hidden size of latent variables inside the RNN unit
  - dropout: dropout rate
  - inner_dropout: the recurrent dropout rate
  - l2_regularization: recurrent l2 regularization rate
  - val_split: the validation split for the input text 
  - optimization: desired optimization algorithm
  - batch_size: batch_size during training
  
### Sampling parameters
  - temperature: controls the randomness of sampling in the text generation phase (in [0,1]; the higher the more random)
  - sampling_length: length of sampled text at the end of training
  - seed: the initial seed required to do sampling (must be at least seq_length long)
  
### Paths
  - SourceFile: file to use in folder SourceText for training
  
  
## Starting the training
Adjust the config.xml to your desires and run main.py. A new folder in Results will be created and the results will be saved there. The results consist of:
  - history.txt: The loss over the epochs
  - loss_plot.png: The loss visualized
  - model.h5: The best model saved (based on validation loss)
  - sampleText.txt: The generated text after training
  
## Visualizing results
In order to get a nice summary of the trained models, run htmlReportGUI.py. A GUI will pop up listing all the different configurations encountered in the Results folder. Choose the configurations you are interested in. After a html file will automatically open in your browser so you can view the results.
  
