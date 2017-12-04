# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 22:43:54 2016

@author: nickv

Main file managing the training.
"""

import os
import numpy as np
from keras.models import load_model
from helper_functions import load_text, load_from_xml
from char_prediction import CharPredictor

def main():
    '''
    Main function:
        - sets up CharPredictor class according to config.xml
        - manages the training
        - executes sampling on trained model
    '''
    # load training configuration from config file    
    params = load_from_xml()
    
    #load text to train from
    text,train_text, val_text, vocab_size, char_to_ix, ix_to_char = \
            load_text(params['source_path'], params['val_split'])
    
    #########################
    # Initialize RNN object #
    #########################
    rnn = CharPredictor(**params, vocab_size = vocab_size, 
                        char_to_ix = char_to_ix, 
                        ix_to_char = ix_to_char)

    ###################
    # BUILD THE MODEL #
    ###################
    model = rnn.build()
    
    ################### 
    # TRAIN THE MODEL #
    ###################
    
    
    # check if model has been interrupted while training (if so continue where we left off)
    if os.path.exists(os.path.join(rnn.result_folder,"history.txt")):
        # find out number of epochs that have been trained already
        history = open(os.path.join(rnn.result_folder,"history.txt")).read()
        train_loss = history.split("Training loss:")[1].split("Validation loss:")[0]
        train_loss = train_loss.lstrip("\n[").rstrip("]\n")
        train_loss = np.fromstring(train_loss, sep = " ")
        rnn.epochs = rnn.epochs - len(train_loss)
        model = load_model(os.path.join(rnn.result_folder, "model.h5"))
    trained_model = rnn.train_generator(model, train_text, val_text)
        
    ###############
    # SAMPLE TEXT #
    ###############
    sampleText = rnn.generate(trained_model, text)
    with open(rnn.sample_path,"w") as f:
        f.write(sampleText)
        
        
if __name__=='__main__':
    main()
    