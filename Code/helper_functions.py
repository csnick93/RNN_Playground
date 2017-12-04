# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 22:42:05 2016

@author: nickv
"""

import numpy as np
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# path to config file
XML_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'Configuration','config.xml')

###########################################################################
def encode_data_labels(data, seq_length,vocab_size,char_to_ix):
    """
    Encodes the data into encoded vectors to do training.
    
    Args:
        data: text in character format
        seq_length: character length of one sample
        vocab_size: alphabet size of text
        char_to_ix: dictionary for character to index mapping
    
    Returns: 
        data in 0-1 vector format, plus corresponding labels encoded
            according to seq-length
    """
    
    # we do not encode last character, as we would not have label for it 
    # (do not know what comes after last character in text)
    x = np.zeros((len(data)-seq_length, seq_length, vocab_size))      
    # need a label for each sequence (len(data)-self.seq_length)-many sequence samples possible
    y = np.zeros((len(data)-seq_length, vocab_size))                       
    
    for s in range(len(data)-seq_length):
        for c in range(seq_length):
            x[s,c,char_to_ix[data[s+c]]] = 1
        y[s,char_to_ix[data[s+seq_length]]] = 1
    return x,y
    
###############################################################################
def load_text(source_path, val_split):
    '''
    Helper function that loads the text from the source text according
    to the validation split.
    
    Args:
        source_path: path to the text file used for training
        val_split: fraction for training and validation split
    '''
    print("Loading text...")
    with open(source_path,'r') as t:
        text = t.read()
        #seperate into training and test text (for validation purposes to determine overfitting/stopping point)
        train_text, val_text = text[:int(len(text)*(1-val_split))], text[int(len(text)*(1-val_split)):]            
        #extract unique character set (the set of different characters we are seeing in the text)
        chars = list(set(text))
        vocab_size = len(chars)
        # create bijective mapping (index,character), so map index onto each character in textfile 
        char_to_ix = { ch:i for i,ch in enumerate(chars) }  
        # using dictionaries, so that we can create vectors encoding the characters
        ix_to_char = { i:ch for i,ch in enumerate(chars) }  
    return text,train_text, val_text, vocab_size, char_to_ix, ix_to_char    
    
 
#########################################
def load_from_xml():
    '''
    Helper function that loads the parameters from config.xml
    
    Returns:
        Dictionary containing all the parameters of config file.
    '''
    tree = ET.parse(XML_CONFIG)
    root = tree.getroot()
    learning_params = root.find('LearningParameters')
    sampling_params = root.find('SamplingParameters')
    paths = root.find('Paths')
    
    params = {}
    
    params['epochs'] = int(learning_params.find('epochs').text)
    params['unit_type'] = learning_params.find('unit_type').text
    params['no_layers'] = int(learning_params.find('no_layers').text)
    params['seq_length'] = int(learning_params.find('seq_length').text)
    params['hidden_size'] = int(learning_params.find('hidden_size').text)
    params['dropout'] = float(learning_params.find('dropout').text)
    params['inner_dropout'] = float(learning_params.find('inner_dropout').text)
    params['l2_regularization'] = float(learning_params.find('l2_regularization').text)
    params['val_split'] = float(learning_params.find('val_split').text)
    params['optimization'] = learning_params.find('optimization').text
    params['batch_size'] = int(learning_params.find('batch_size').text)
    
    params['temperature'] = float(sampling_params.find('temperature').text)
    params['sampling_length'] = int(sampling_params.find('sampling_length').text)
    params['seed'] = sampling_params.find('seed').text
    
    params['source_path'] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'SourceText',paths.find('SourceFile').text)
    return params

##############################################################################
def save_loss_plot(train_loss,val_loss,save_path):
    '''
    Create loss plot for training and validation loss.
    
    Args:
        train_loss: training loss np array
        val_loss: validation loss np array
        save_path: path where plot is saved
    '''
    plt.plot(train_loss,"r", label = "Training loss")
    plt.plot(val_loss,"b", label = "Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    
##############################################################################
    
def save_loss_data(time,train_loss, val_loss, save_path):
    '''
    Save loss and required training time.
    
    Args:
        time: required training time
        train_loss: training loss np array
        val_loss: validation loss np array
        save_path: path where plot is saved
    '''
    with open(save_path, "w") as f:
        f.write("Training took: %f seconds\n"%time)
        f.write("\nTraining loss:\n")
        f.write(np.array_str(train_loss))
        f.write("\nValidation loss:\n")
        f.write(np.array_str(val_loss))
        
##############################################################################
def collect_options(result_folder):
    '''
    Helper function that collects all the options for 
        - unit-type
        - source text
        - number of layers
        - epochs of training
    based on the trained models in the Result folder.
    
    Args:
        result_folder: path to result folder where trained models lie
    
    Returns:
        list of all the options mentioned above
    '''
    source_texts = []
    unit_types = []
    number_layers = []
    epochs_training = []
    for model in os.listdir(result_folder):
        model_path = os.path.join(result_folder,model)
        if 'model_config.txt' in os.listdir(model_path):
            with open(os.path.join(model_path,'model_config.txt'),'r') as info:
                info_txt = info.read()
                unit_types.append(info_txt.split('unit_type: ')[1].split('\n')[0])
                source_texts.append(os.path.basename(info_txt.split('source_path: ')[1].split('\n')[0]))
                number_layers.append(info_txt.split('no_layers: ')[1].split('\n')[0])
                epochs_training.append(info_txt.split('epochs: ')[1].split('\n')[0])
    
    source_texts = list(set(source_texts))
    unit_types = list(set(unit_types))
    number_layers = [str(e) for e in sorted([int(no) for no in list(set(number_layers))])]
    epochs_training = [str(e) for e in sorted([int(e) for e in list(set(epochs_training))])]
                
    return source_texts, unit_types, number_layers, epochs_training
                