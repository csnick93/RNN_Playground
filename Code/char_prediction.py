# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 13:59:54 2016

@author: nickv

Character Prediction class.
"""


import os
import random
import numpy as np
import time
from keras.models import Sequential, load_model
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from helper_functions import save_loss_plot, save_loss_data

class CharPredictor:
   
    def __init__(self,epochs, unit_type, no_layers, seq_length,
                 hidden_size, dropout, inner_dropout, l2_regularization,
                 val_split, optimization, batch_size,
                 temperature, sampling_length, seed, source_path,
                 vocab_size, char_to_ix, ix_to_char):
        """
        Args:
            epochs : number of epochs we are training the model
            unit_type: choice between LSTM,GRU, and BasicRNN
            no_layers: number of layers EXCLUDING input layer and output 
                                layer
            seq_length: seq-length of input we need for training and later also
                                sampling
            hidden_size: size of RNN internal representation
            dropout: dropout rate
            inner_dropout: recurrent dropout rate
            l2_regularization: l2 regularization rate
            val_split: ratio of validation data of whole text
            optimization: type of optimization method
            batch_size: batch size for training
            temperature: the higher the temperature, the more random the 
                            sampling function will get
            sampling_length: the number of characters we are predicting in the 
                                generate function
            seed: initial sequence of characters for sampling (must be at least 
                                seq_length long)
            source_path: where to get text_file from for learning
            vocab_size: size of alphabet (necessary for 1-hot encoding)
            char_to_ix, ix_to_char: dictionaries encoding bijective index to character mapping
        """
        
        # Training parameters
        self.epochs = epochs
        self.unit_type = unit_type
        self.no_layers = no_layers
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.inner_dropout = inner_dropout
        self.l2_regularization = l2_regularization  
        self.val_split = val_split
        self.optimization = optimization
        self.batch_size = batch_size
        
        # Sampling parameters
        self.temperature = temperature
        self.sampling_length = sampling_length
        self.optimization = optimization
        self.seed = seed
        
        # Paths
        self.source_path = source_path
        self.result_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'Results')
       
        # Source Text information  
        self.vocab_size = vocab_size
        self.char_to_ix = char_to_ix
        self.ix_to_char = ix_to_char
        
        
        # create folder for results of the configuration
        self.result_folder = os.path.join(self.result_path, 'model_%i'%len(os.listdir(self.result_path)))
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        self.model_path = os.path.join(self.result_folder,"model.h5")
        self.history_path = os.path.join(self.result_folder,"history.txt")
        self.sample_path = os.path.join(self.result_folder,"sampleText.txt")
        
        # save configuration
        with open(os.path.join(self.result_folder,'model_config.txt'),'w') as info:
            info.write("Experiment configuration:\n")
            for k in self.__dict__.keys():
                info.write('\t'+k +": "+ str(self.__dict__[k])+"\n")

    ###########################################################################
    # BUILD MODEL #
    ############### 
    def build(self):
        '''
        Member function responsible for building the RNN model for later 
        training.
        
        Returns:
            The built RNN keras model.
        '''
        
        print("Building model...")
        model = Sequential()
        #########
        # LSTMS #
        #########
        if(self.unit_type == "LSTM"):
            if(self.no_layers == 1):
                model.add(LSTM(self.hidden_size, return_sequences=False, input_shape=(self.seq_length, self.vocab_size), 
                               recurrent_regularizer=l2(self.l2_regularization), 
                                recurrent_dropout=self.inner_dropout))
            else:
                model.add(LSTM(self.hidden_size, return_sequences=True, input_shape=(self.seq_length, self.vocab_size),
                               recurrent_regularizer=l2(self.l2_regularization), 
                                recurrent_dropout=self.inner_dropout))
                
            
                
            for l in range(self.no_layers-2):
                model.add(LSTM(self.hidden_size, return_sequences=True,init='glorot_uniform', inner_init='orthogonal',
                               forget_bias_init='one', activation='tanh', inner_activation='hard_sigmoid', 
                               recurrent_regularizer=l2(self.l2_regularization), 
                               b_regularizer=None, recurrent_dropout=self.inner_dropout))
                
                
            if (self.no_layers > 1):    
                model.add(LSTM(self.hidden_size, return_sequences=False,init='glorot_uniform', inner_init='orthogonal', 
                               forget_bias_init='one', activation='tanh', inner_activation='hard_sigmoid', 
                               recurrent_regularizer=l2(self.l2_regularization), 
                               b_regularizer=None, recurrent_dropout=self.inner_dropout))
    
               
                    
        #########################
        # GATED RECURRENT UNITS #
        #########################
        elif(self.unit_type == "GRU"):
            if(self.no_layers==1):
                model.add(GRU(self.hidden_size, return_sequences=False, input_shape=(self.seq_length, self.vocab_size),
                              recurrent_regularizer=l2(self.l2_regularization),
                              recurrent_dropout=self.inner_dropout))
            else:
                model.add(GRU(self.hidden_size, return_sequences=True, input_shape=(self.seq_length, self.vocab_size),
                              recurrent_regularizer=l2(self.l2_regularization), 
                              recurrent_dropout=self.inner_dropout))
            
            
                
            for l in range(self.no_layers-2):
                model.add(GRU(self.hidden_size, return_sequences=True,init='glorot_uniform', inner_init='orthogonal',
                              activation='tanh', inner_activation='hard_sigmoid', recurrent_regularizer=l2(self.l2_regularization),
                                recurrent_dropout=self.inner_dropout))            
                
                
            
            if (self.no_layers > 1):        
                model.add(GRU(self.hidden_size, return_sequences=False,init='glorot_uniform', inner_init='orthogonal',
                                  activation='tanh', inner_activation='hard_sigmoid', recurrent_regularizer=l2(self.l2_regularization),
                                 recurrent_dropout=self.inner_dropout))
                
                
                    
        ###############
        # SIMPLE RNNs #
        ###############
        else:
            if(self.no_layers == 1):
                model.add(SimpleRNN(self.hidden_size, return_sequences = False,input_shape=(self.seq_length, self.vocab_size),
                                    recurrent_regularizer=l2(self.l2_regularization),
                                    recurrent_dropout=self.inner_dropout))
            else:
                model.add(SimpleRNN(self.hidden_size, return_sequences = True,input_shape=(self.seq_length, self.vocab_size),
                                    recurrent_regularizer=l2(self.l2_regularization),
                                    recurrent_dropout=self.inner_dropout))
            
           
               
            for l in range(self.no_layers-2):
                model.add(SimpleRNN(self.hidden_size, return_sequences = True,
                                    recurrent_regularizer=l2(self.l2_regularization),
                                    recurrent_dropout=self.inner_dropout))
                
                model.add(Dropout(self.dropout))
            if (self.no_layers > 1):
                model.add(SimpleRNN(self.hidden_size,
                                    recurrent_regularizer=l2(self.l2_regularization),
                                    recurrent_dropout=self.inner_dropout))
                
            
        #####################################################   
        # add Dense layer at the end to produce probabilities of chars
        model.add(Dense(self.vocab_size))
        model.add(Activation('softmax'))
        return model
        
    
    ###########################################################################
    # TRAIN MODEL #
    ###############
    def train_generator(self,model, train_text, val_text):
        '''
        Member function variant of training, using the fit_generator method.
        This is necessary when the text is too big to process it as a whole.
        
        Args:
            model: the built model to train
            train_text: text as whole on which shall be trained, the splitting
                        into sequences is done by the data generator
            val_text: text for validation
            
        Returns:
            The best performing model according to the validation loss.
        '''
        
        def data_generator(data, batch_size,seq_length, char_to_ix, vocab_size):
            '''
            Produces batches of size |batch_size| infinitely -> will go over text repeatedly.
            Always steps one character forward with each succeeding generated sequence 
            (so we produce overlap of seq_length-1).
            
            Args:
                data: the whole text string to be used for training
                batch_size: batch_size for the training
                seq_length: length of a text sample
                char_to_ix: dictionary to map char-> index
                vocab_size: number of different occurring characters in text
                
            Returns:
                Training/validation batch x,y
            '''
            counter = 0
            while(True):
                if ((counter+1)*batch_size+seq_length < len(data)):
                    start_char = counter*batch_size
                    counter +=1 
                else:
                    start_char = 0
                    counter = 0
                # encode chars in sequence for the |batch-size| sequences
                x = np.zeros((batch_size,seq_length,vocab_size))
                # encode correct output char for the |batch-size| sequences
                y = np.zeros((batch_size, vocab_size),dtype = np.int32)
                
                # encode |batch_size| sequences of length SEQ_LENGTH 
                #(for each new sequence, move STEP many characters ahead (produce overlap))
                for n in range(0,batch_size):
                    ptr = start_char+n
                    for i in range(seq_length):
                        x[n,i,char_to_ix[data[ptr+i]]] = 1.
                    y[n,char_to_ix[data[ptr+seq_length]]] = 1
                yield x, y
        
        
        
        print("Starting trainig...")
        start_time = time.time()
        model.compile(loss = "categorical_crossentropy",optimizer = self.optimization)
        saveBestModel = ModelCheckpoint(self.model_path,monitor = "val_loss", verbose = 1, save_best_only = True)
        no_samples_per_epoch = int((len(train_text)-self.seq_length)/self.batch_size)    # calculate how many samples can be created out of text
        no_val_samples_per_epoch = (len(val_text)-self.seq_length-1)
        
        
        hist = model.fit_generator(data_generator(train_text, self.batch_size, self.seq_length, self.char_to_ix, self.vocab_size), 
                                   no_samples_per_epoch, self.epochs,
                                   callbacks = [saveBestModel], 
                                   validation_data = data_generator(val_text, self.batch_size, self.seq_length, self.char_to_ix, self.vocab_size),
                                   validation_steps = no_val_samples_per_epoch)
        stop_time = time.time()

        # save training history
        np_hist_loss = np.array(hist.history["loss"])
        np_hist_val_loss = np.array(hist.history["val_loss"])
        save_loss_plot(np_hist_loss, np_hist_val_loss, os.path.join(self.result_folder,'loss_plot.png'))
        save_loss_data(stop_time-start_time,np_hist_loss, np_hist_val_loss, os.path.join(self.result_folder,'history.txt'))
        # get best model
        model = load_model(self.model_path)
        print("Done with training")
        return model
    
    ###########################################################################
    # SAMPLE TEXT #
    ###############
    def generate(self,model,text):
        """
        Member function to do generation of sample text based on trained model
        and some text to start the sampling process.
        
        Args:
            model: trained model to sample from
            text: char-text (not encoded), just needed to start sampling
            
        Returns:
            Generated text.
        """    
        
        def sample(probs, temperature):
            '''
            Helper function, which samples an index from a vector of probabilities.
            
            Args:
                probs: probability vector for the next characters
                temperature: rescales the probabilities (controls randomness of sampling)
                        temp = 1: probabilities stay the same
                        temp = 0: max probability is 1 and rest is 0
                        
            Returns:
                Sampled index corresponding to a character to be sampled next.
            '''
            a = np.log(probs)/temperature
            a = np.exp(a)/np.sum(np.exp(a))
            # due to numerical stability reasons when drawing index, subtract an epsilon
            eps = 1e-8
            a = a - ((np.sum(a)-1)/len(a)) - eps
            return np.argmax(np.random.multinomial(1, a, 1))
        
        
        
        def sequence_encoder(data, vocab_size, char_to_ix):
            '''
            Encode a sequence of chars into 0-1 vectors.
            
            Args:
                data: Samples of string text to be one-hot encoded
                vocab_size: number of characters in text -> determines length
                            of one-hot vector
                char_to_ix: dictionary char -> index
                
            Returns:
                data encoded as one-hot vectors
            '''
            # encode chars in sequence for the |batch-size| sequences
            x = np.zeros((len(data),vocab_size))    
            # encode |batch_size| sequences of length SEQ_LENGTH (
            # for each new sequence, move STEP many characters ahead (produce overlap))
            for i in range(0, len(data)):
                x[i,char_to_ix[data[i]]] = 1.
            return x
        
        
        print("Generating sample text...")
        predicate=lambda x: len(x) < self.sampling_length
        # if seed does not meet requirements, randomly choose chunk of text from f
        if self.seed is None or len(self.seed) < self.seq_length:
            start_idx = random.randint(0, len(text) - self.seq_length - 1)
            self.seed = text[start_idx:start_idx + self.seq_length]
        
        sentence = self.seed[:self.seq_length]
        generated_text = sentence        # here our predicted string is being generated
    
        while predicate(generated_text):
            # generate the input tensor
            # from the last max_len characters generated so far
            # encode the input sequence
            x = np.zeros((1,self.seq_length,self.vocab_size))
            x[0] = sequence_encoder(sentence[-self.seq_length:], self.vocab_size,
                                     self.char_to_ix)
    
            # this produces a probability distribution over characters
            probs = model.predict(x, verbose=0)[0]
    
            # sample the character to use based on the predicted probabilities
            next_idx = sample(probs, self.temperature)
            next_char = self.ix_to_char[next_idx]
    
            generated_text += next_char
            sentence = sentence[1:] + next_char
        return generated_text

