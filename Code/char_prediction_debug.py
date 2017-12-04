# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 08:38:57 2016

@author: nickv
"""

# -*- coding: utf-8 -*-
"""
COMMANDS:
    - making theano backend: export KERAS_BACKEND=theano
    - enabling gpu device: THEANO_FLAGS=device=gpu python keras_char_prediction.py


Created on Sun Nov 20 13:59:54 2016

@author: nickv

Character Prediction using Keras
"""

"""
TODO:
    encoding of character is gen_data does not work properly

""" 

import os
import random
import numpy as np
import time
from subprocess import call
from keras.models import Sequential, load_model
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l2

location = 0       # 0: cipPool, 1: windows pc, else: linux pc
class CharPredictor:
    """
    epochs : number of epochs we are training the model
    seq_length: seq-length of input we need for training and later also sampling
    train_batch_size: number of sequences to be fed in before updating parameters
    temperature: the higher the temperature, the more random the sampling function will get
    predict_length: the number of characters we are predicting in the generate function
    loss: type of loss function
    optimization: type of optimization method
    seed: initial sequence of characters for sampling (must be at least seq_length long)
    number_of_layers: number of layers EXCLUDING input layer and output layer 
    unit_type: choice between LSTM,GRU, and BasicRNN
    dropout: wether to add a dropout layer after each hidden layer
    dropout_p: if dropout is used, what dropout-percentage to use
    patience: number of epochs to wait for improvement
    delta: necessary improvement to be considered an improvement
    source_path: where to get text_file from for learning
    model_path: where to save best model
    sample_path: where to save sampled text
    val_split: ratio of validation data of whole text
    data_batch_size: how big should batches be in data generator (only relevant for big text)
    """
    def __init__(self,epochs, seq_length,source_path,result_path,batch_size = 32, temperature = .35, predict_length = 1000,
                 loss = "categorical_crossentropy", optimization = "rmsprop", seed = None, 
                 number_of_layers = 2, unit_type = "GRU", dropout = True, dropout_p = 0.5, 
                 earlyStopping = False,patience = 100, delta = 1e-3, hidden_size = 512,
                 bigText = False, val_split = .1, data_batch_size = 50):
        self.epochs = epochs
        self.seq_length = seq_length
        self.batch_size= batch_size
        self.temperature = temperature
        self.predict_length = predict_length
        self.loss = loss
        self.optimization = optimization
        self.seed = seed
        self.no_layers = number_of_layers
        self.unit_type = unit_type
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.earlyStopping = earlyStopping
        self.patience = patience
        self.delta = delta
        self.hidden_size = hidden_size
        self.source_path = source_path
        self.result_path = result_path
        self.inner_dropout = 0.3
        self.regularization = 0.1
        # create folder for results of the configuration
        if location == 1:
            sourceFile = source_path.split("\\")[-1].rstrip(".txt")
        else:
            sourceFile = source_path.split("/")[-1].rstrip(".txt")
        self.result_folder = os.path.join(result_path,sourceFile,"Folder_"+str(number_of_layers)+"_"+str(epochs)+"_"+str(seq_length)+"_"+unit_type+"_"+
                                       str(dropout)+"_"+str(dropout_p)+ "_"+str(earlyStopping)+"_"+str(patience)+"_"+str(hidden_size)+"_"
                                       +str(self.inner_dropout)+"_"+str(self.regularization))
        if not os.path.exists(os.path.join(result_path,sourceFile)):
            os.mkdir(os.path.join(result_path,sourceFile))
        if not os.path.exists(self.result_folder):
            os.mkdir(self.result_folder)
        self.model_path = os.path.join(self.result_folder,"model.h5")
        self.history_path = os.path.join(self.result_folder,"history.txt")
        self.sample_path = os.path.join(self.result_folder,"sampleText.txt")
        
        #initialized in loadText
        self.data_size = 0
        self.vocab_size = 0
        self.chars = None
        self.char_to_ix = None
        self.ix_to_char = None
        self.data_batch_size = data_batch_size

    ###############################################################################
    # BUILD MODEL #
    ############### 
    def build(self):
        print("====BUILDING MODEL====")
        model = Sequential()
        #################################################
        # LSTMS #
        #########
        if(self.unit_type == "LSTM"):
            if(self.no_layers == 1):
                model.add(LSTM(self.hidden_size, return_sequences=False, input_shape=(self.seq_length, len(self.chars)), 
                               W_regularizer=l2(self.regularization), U_regularizer=l2(self.regularization), dropout_W=self.inner_dropout, dropout_U=self.inner_dropout))
            else:
                model.add(LSTM(self.hidden_size, return_sequences=True, input_shape=(self.seq_length, len(self.chars)),
                               W_regularizer=l2(self.regularization), U_regularizer=l2(self.regularization), dropout_W=self.inner_dropout, dropout_U=self.inner_dropout))
            if(self.dropout):
                model.add(Dropout(self.dropout_p))
                
            for l in range(self.no_layers-2):
                model.add(LSTM(self.hidden_size, return_sequences=True,init='glorot_uniform', inner_init='orthogonal',            # must set return_sequences = True, if we want to stack LSTM layers
                               forget_bias_init='one', activation='tanh', inner_activation='hard_sigmoid', 
                               W_regularizer=l2(self.regularization), U_regularizer=l2(self.regularization), b_regularizer=None, dropout_W=self.inner_dropout, dropout_U=self.inner_dropout))            
                if(self.dropout):            
                    model.add(Dropout(self.dropout_p))
            if (self.no_layers > 1):    
                model.add(LSTM(self.hidden_size, return_sequences=False,init='glorot_uniform', inner_init='orthogonal', 
                               forget_bias_init='one', activation='tanh', inner_activation='hard_sigmoid', 
                               W_regularizer=l2(self.regularization), U_regularizer=l2(self.regularization), b_regularizer=None, dropout_W=self.inner_dropout, dropout_U=self.inner_dropout))
                if(self.dropout):
                    model.add(Dropout(self.dropout_p))       
        ###################################################
        # GATED RECURRENT UNITS #
        #########################
        elif(self.unit_type == "GRU"):
            if(self.no_layers==1):
                model.add(GRU(self.hidden_size, return_sequences=False, input_shape=(self.seq_length, len(self.chars)),
                              W_regularizer=l2(self.regularization), U_regularizer=l2(self.regularization),dropout_W=self.inner_dropout, dropout_U=self.inner_dropout))
            else:
                model.add(GRU(self.hidden_size, return_sequences=True, input_shape=(self.seq_length, len(self.chars)),
                              W_regularizer=l2(self.regularization), U_regularizer=l2(self.regularization), dropout_W=self.inner_dropout, dropout_U=self.inner_dropout))
            if(self.dropout):
                model.add(Dropout(self.dropout_p))
                
            for l in range(self.no_layers-2):
                model.add(GRU(self.hidden_size, return_sequences=True,init='glorot_uniform', inner_init='orthogonal',                 # must set return_sequences = True, if we want to stack GRU layers
                              activation='tanh', inner_activation='hard_sigmoid', W_regularizer=l2(self.regularization), U_regularizer=l2(self.regularization)
                              , b_regularizer=None, dropout_W=self.inner_dropout, dropout_U=self.inner_dropout))            
                if(self.dropout):
                    model.add(Dropout(self.dropout_p))
            
            if (self.no_layers > 1):        
                model.add(GRU(self.hidden_size, return_sequences=False,init='glorot_uniform', inner_init='orthogonal',                 # must set return_sequences = True, if we want to stack GRU layers
                                  activation='tanh', inner_activation='hard_sigmoid', W_regularizer=l2(self.regularization), U_regularizer=l2(self.regularization)
                                  , b_regularizer=None, dropout_W=self.inner_dropout, dropout_U=self.inner_dropout))
                if(self.dropout):
                    model.add(Dropout(self.dropout_p))
        else:
            ###############
            # SIMPLE RNNs #
            ###############
            # if we add only one hidden layer we do not want sequences to be returned
            if(self.no_layers == 1):
                model.add(SimpleRNN(self.hidden_size, return_sequences = False,input_shape=(self.seq_length, len(self.chars)),
                                    W_regularizer=l2(self.regularization), U_regularizer=l2(self.regularization),dropout_W=self.inner_dropout, dropout_U=self.inner_dropout))
            else:
                model.add(SimpleRNN(self.hidden_size, return_sequences = True,input_shape=(self.seq_length, len(self.chars)),
                                    W_regularizer=l2(self.regularization), U_regularizer=l2(self.regularization),dropout_W=self.inner_dropout, dropout_U=self.inner_dropout))
            if(self.dropout):
                model.add(Dropout(self.dropout_p))
               
            for l in range(self.no_layers-2):
                model.add(SimpleRNN(self.hidden_size, return_sequences = True,
                                    W_regularizer=l2(self.regularization), U_regularizer=l2(self.regularization),dropout_W=self.inner_dropout, dropout_U=self.inner_dropout))            
                if(self.dropout):
                    model.add(Dropout(self.dropout_p))
            if (self.no_layers > 1):        
                model.add(SimpleRNN(self.hidden_size,
                                    W_regularizer=l2(self.regularization), U_regularizer=l2(self.regularization),dropout_W=self.inner_dropout, dropout_U=self.inner_dropout))
                if(self.dropout):
                    model.add(Dropout(self.dropout_p))
            
        #####################################################   
        # add Dense layer at the end to produce probabilities of chars
        model.add(Dense(len(self.chars)))
        model.add(Activation('softmax'))
        return model
        
    ###############################################################################
    
    ############
    # TRAINING #
    ############
    """
    "automatic" training done by keras implemented function
    """       
    # NOTE: There will be a warning because one epoch will not use up input_data completely
    
    def train(self, model, X_train, y_train, X_val, y_val):
        print("===STARTING TRAINING====")
        start_time = time.time()
        model.compile(loss = self.loss,optimizer = self.optimization)
        saveBestModel = ModelCheckpoint(self.model_path,monitor = "val_loss", verbose = 1, save_best_only = True)
        if(self.earlyStopping):
            stopping = EarlyStopping("val_loss", min_delta = self.delta, patience=self.patience)
            hist = model.fit(X_train,y_train, batch_size = self.batch_size, nb_epoch = self.epochs, callbacks = [saveBestModel, stopping],
                      validation_data = (X_val,y_val))
        else:
            hist = model.fit(X_train,y_train, batch_size = self.batch_size, nb_epoch = self.epochs, callbacks = [saveBestModel],
                      validation_data = (X_val,y_val))
        stop_time = time.time()

        # save training history
        np_hist_loss = np.array(hist.history["loss"])
        np_hist_val_loss = np.array(hist.history["val_loss"])
        with open(self.history_path, "w") as f:
            f.write("Training took: "+ str(stop_time-start_time) + " seconds\n")
            f.write("\nTraining loss:\n")
            f.write(np.array_str(np_hist_loss))
            f.write("\nValidation loss:\n")
            f.write(np.array_str(np_hist_val_loss))
        # get best model
        model = loadModel(self.model_path)
        print("=== DONE WITH TRAINING ===")
        return model
    
    #############################################################################
    """
    training using model.fit_generator
    """
    def train_generator(self,model, train_text, val_text):
        print("===STARTING TRAINING====")
        start_time = time.time()
        model.compile(loss = self.loss,optimizer = self.optimization)
        saveBestModel = ModelCheckpoint(self.model_path,monitor = "val_loss", verbose = 1, save_best_only = True)
        no_samples_per_epoch = (len(train_text)-self.seq_length)    # calculate how many samples can be created out of text
        no_val_samples_per_epoch = (len(val_text)-self.seq_length) 
        if(self.earlyStopping):
            stopping = EarlyStopping("val_loss", min_delta = self.delta, patience=self.patience)
            hist = model.fit_generator(self.data_generator(train_text, self.data_batch_size), no_samples_per_epoch, self.epochs,
                                       callbacks = [saveBestModel, stopping], validation_data = self.data_generator(val_text, self.data_batch_size),
                                        nb_val_samples = no_val_samples_per_epoch)
        else:
            hist = model.fit_generator(self.data_generator(train_text, self.data_batch_size), no_samples_per_epoch, self.epochs,
                                       callbacks = [saveBestModel], validation_data = self.data_generator(val_text, self.data_batch_size),
                                        nb_val_samples = no_val_samples_per_epoch)
        stop_time = time.time()

        # save training history
        np_hist_loss = np.array(hist.history["loss"])
        np_hist_val_loss = np.array(hist.history["val_loss"])
        with open(self.history_path, "w") as f:
            f.write("Training took: "+ str(stop_time-start_time) + " seconds\n")
            f.write("\nTraining loss:\n")
            f.write(np.array_str(np_hist_loss))
            f.write("\nValidation loss:\n")
            f.write(np.array_str(np_hist_val_loss))
        # get best model
        model = loadModel(self.model_path)
        print("=== DONE WITH TRAINING ===")
        return model
    ###############################################################################
    # SAMPLE TEXT #
    ###############
        
    """
    model: trained model
    temperature: randomness of sampling
    text: char-text (not encoded), just needed for seed generation
    predicate: determines when we are done predicting 
    """    
        
    def generate(self,model,text):
        print("===GENERATING SAMPLE TEXT===")
        predicate=lambda x: len(x) < self.predict_length
        # if seed does not meet requirements, randomly choose chunk of text from f
        if self.seed is None or len(self.seed) < self.seq_length:
            start_idx = random.randint(0, len(text) - self.seq_length - 1)
            seed = text[start_idx:start_idx + self.seq_length]
        
        sentence = seed[:self.seq_length]
        generated = sentence        # here our predicted string is being generated
    
        while predicate(generated):
            # generate the input tensor
            # from the last max_len characters generated so far
            # encode the input sequence
            x = np.zeros((1,self.seq_length,self.vocab_size))
            x[0] = self.sequence_encoder(sentence[-self.seq_length:])
    
            # this produces a probability distribution over characters
            probs = model.predict(x, verbose=0)[0]
    
            # sample the character to use based on the predicted probabilities
            next_idx = self.sample(probs)
            next_char = self.ix_to_char[next_idx]
    
            generated += next_char
            sentence = sentence[1:] + next_char
        return generated


    ################################################################################
    """
    the temperature rescales the probabilities (if temp = 1: probabilities stay the same, if temp = 0: max probability is 1 and rest is 0)
    """
    def sample(self,probs):
        # the higher the temperature, the more likely an unlikely character will be chosen
        """samples an index from a vector of probabilities"""
        a = np.log(probs)/self.temperature
        a = np.exp(a)/np.sum(np.exp(a))
        a = a - ((np.sum(a)-1)/len(a)) - 1e-8         # so that we have a sum of slightly below one in array to represent a pdf (impossible to have exactly sum of 1 due to numerical instabilities, therefore try to get slightly below)
        return np.argmax(np.random.multinomial(1, a, 1))
        
    ###############################################################################
    #  DATA ENCODER   #
    ###################
    # encode a sequence of chars into 0-1 vectors
    def sequence_encoder(self,data):
              
        x = np.zeros((len(data),self.vocab_size))    # encode chars in sequence for the |batch-size| sequences
        # encode |batch_size| sequences of length SEQ_LENGTH (for each new sequence, move STEP many characters ahead (produce overlap))
        for i in range(0, len(data)):
            x[i,self.char_to_ix[data[i]]] = 1.
        return x
    
    ###############################################################################
    #  DATA GENERATOR #
    ###################
    # generate batches infinitely: necessary if we want to use model.fit_generator
    def data_generator(self,data, batch_size,return_target=True):
        '''
        This function is similar to gen_data. Yet, it produces batches of size |batch_size|
        infinitely -> will go over text repeatedly
        -> we always step one character forward with each succeeding generated sequence (so we produce overlap of seq_length-1)
        '''
        counter = 0
        while(True):
            if ((counter+1)*batch_size+self.seq_length < len(data)):
                start_char = counter*batch_size
                counter +=1 
            else:
                start_char = 0
                counter = 0
                    
            x = np.zeros((batch_size,self.seq_length,self.vocab_size))    # encode chars in sequence for the |batch-size| sequences
            y = np.zeros((batch_size, self.vocab_size),dtype = np.int32)              # encode correct output char for the |batch-size| sequences
            
            # encode |batch_size| sequences of length SEQ_LENGTH (for each new sequence, move STEP many characters ahead (produce overlap))
            for n in range(0,batch_size):
                ptr = start_char+n
                for i in range(self.seq_length):
                    x[n,i,self.char_to_ix[data[ptr+i]]] = 1.
                if(return_target):
                    y[n,self.char_to_ix[data[ptr+self.seq_length]]] = 1
            
            yield x, y
    
    ###########################################################################
    # COMPUTE RECALL AND PRECISION #
    ################################
    # compute recall and precision based on validation set
    # 
    #def computeRecallPrecision(self, model, X_val, y_val):

###########################################################################
"""
decode for debugging purposes

X: one sample sentence
ix_to_char: dictionary for decoding
"""

def decodeText(X, ix_to_char):
    sentence = ""
    for vec in X:
        sentence += ix_to_char[np.where(vec==1)[0][0]]
    return sentence
###########################################################################
"""
data: text in character format
return: data in 0-1 vector format, plus corresponding labels encoded
        according to seq-length
"""
def encode_data_labels(data, seq_length,vocab_size,char_to_ix):
    x = np.zeros((len(data)-seq_length, seq_length, vocab_size))      # we do not encode last character, as we would not have label for it (do not know what comes after last character in text)
    y = np.zeros((len(data)-seq_length, vocab_size))                       # need a label for each sequence (len(data)-self.seq_length)-many sequence samples possible
    
    for s in range(len(data)-seq_length):
        for c in range(seq_length):
            x[s,c,char_to_ix[data[s+c]]] = 1
        y[s,char_to_ix[data[s+seq_length]]] = 1
    return x,y
    
 ###############################################################################
# LOAD AND SEPERATE TEXT, ENCODE CHARACTERS #
#############################################
#load up the text
"""
how much of text should be used as validation data
"""
def loadText(source_path, val_split):
    print("===LOADING TEXT===")
    text = open(source_path,'r').read()
    train_text, val_text = text[:int(len(text)*(1-val_split))], text[int(len(text)*(1-val_split)):]            #seperate into training and test text (for validation purposes to determine overfitting/stopping point)
    #extract unique character set (the set of different characters we are seeing in the text)
    chars = list(set(text))
    vocab_size = len(chars)
    char_to_ix = { ch:i for i,ch in enumerate(chars) }  # create bijective mapping (index,character), so map index onto each character in textfile 
    ix_to_char = { i:ch for i,ch in enumerate(chars) }  # using dictionaries, so that we can create vectors encoding the characters
    #print("vocab_size: " + str(vocab_size))        
    return text,train_text, val_text, vocab_size, char_to_ix, ix_to_char,chars
    
################################################################################
# SAVE THE MODEL #
##################
def save(model, path = r"C:\Users\nickv\Documents\WS1617\DL\Coding\char_pred_model.h5"):
    model.save(path) 

################################################################################
# LOAD A MODEL   #
##################
def loadModel(path = r"C:\Users\nickv\Documents\WS1617\DL\Coding\char_pred_model.h5"):
    return load_model(path)          

    
###############################################################################
# Check if configuration has already been trained #
##################################################
def alreadyProcessed(source, result_folder,result_path):
    sourceFile = os.path.basename(source)
    logfile_path = os.path.join(result_path, sourceFile.rstrip(".txt")+"_log.txt")
    folderName = os.path.basename(result_folder)
    f = open(logfile_path)
    processedCases = f.read()
    if folderName in processedCases:
        f.close()
        return True
    else:
        f.close()
        return False
        
####################################################################    
################
# TEST NETWORK #
################
def main(sample = False):
    """
    def __init__(self,epochs, seq_length,batch_size = 32, temperature = .35, predict_length = 1000,
                 loss = "categorical_crossentropy", optimization = "rmsprop", seed = None,
                 number_of_layers = 2, unit_type = "GRU", dropout = False, dropout_p = 0.2, 
                 earlyStopping = False,patience = 100, delta = 1e-3, hidden_size = 512,
                 source_path = r"C:\Users\nickv\Documents\WS1617\DL\Coding\CharacterPrediction\SourceText\RandomTestText.txt",
                 model_path = r"C:\Users\nickv\Documents\WS1617\DL\Coding\CharacterPrediction\Models",
                 sample_path = r"C:\Users\nickv\Documents\WS1617\DL\Coding\CharacterPrediction\SampleText", 
                 bigText = False, val_split = .1, data_batch_size = 50):
    """
    
    if location == 0:
        sourcePaths = [r"/home/cip/2014/na02zijy/Documents/WS1617/DL/SourceText/bigTextfile.txt"] 
                       #r"/home/cip/2014/na02zijy/Documents/WS1617/DL/SourceText/accumulatedCode.txt"]
    elif location == 1:
        sourcePaths = [r"C:\Users\nickv\Documents\WS1617\DL\Coding\CharacterPrediction\SourceText\bigTextfile.txt"]
                       #r"C:\Users\nickv\Documents\WS1617\DL\Coding\CharacterPrediction\SourceText\accumulatedCode.txt"]
    else:
        sourcePaths = [r"/home/nick/Documents/DeepLearning/CharPrediction/bigTextfile.txt",
                       r"/home/nick/Documents/DeepLearning/CharPrediction/accumulatedCode.txt"]
    epochs = [3]
    unit_type = ["GRU"]
    layers = [1]
    seq_length = [20]
    hiddenSize = [512]
    dropout = [True]
    bigText = False             # flag to indicate whether we are dealing with a big text that would result in memory error
    val_split = .1              # validation split of data
    for source in sourcePaths:
        for seq in seq_length:
            text,train_text, val_text, vocab_size, char_to_ix, ix_to_char,chars = loadText(source, val_split)
            ##
            # Try encoding the text, however, if text is large, there will be a memory error.
            #   In that case, we need to use the method model.fit_generator
            try:
                X_train, y_train = encode_data_labels(train_text,seq, vocab_size, char_to_ix)
                X_val, y_val = encode_data_labels(val_text,seq, vocab_size, char_to_ix)
            except MemoryError:
                bigText = True
            for l in layers:
                for e in epochs:
                    for h in hiddenSize:
                        for unit in unit_type:
                            for drop in dropout:
                                print("Model:"+"\nSourceText:" + source + "\nEpochs: "+str(e)+ "\nLayers: "+str(l) + 
                                      "\nHiddenSize: "+str(h) + "\nSequence Length: "+ str(seq))
                                #########################
                                # initialize RNN object #
                                #########################
                                if location == 0:
                                    rnn = CharPredictor(epochs = e,seq_length = seq, unit_type= unit,
                                                source_path = source,dropout = drop,
                                                result_path = r"/proj/ciptmp/na02zijy/Results",
                                                number_of_layers= l, hidden_size = h)
                                elif location == 1:
                                    rnn = CharPredictor(epochs = e,seq_length = seq, unit_type= unit,
                                                source_path = source, number_of_layers=l, dropout = drop,
                                                result_path = r"C:\Users\nickv\Documents\WS1617\DL\Coding\CharacterPrediction\Results",
                                                hidden_size = h)
                                else:
                                    rnn = CharPredictor(epochs = e,seq_length = seq, unit_type= unit,
                                                source_path = source, number_of_layers=l, dropout = drop,
                                                result_path = r"/home/nick/Documents/DeepLearning/CharPrediction/Results",
                                                hidden_size = h)
                                rnn.vocab_size = vocab_size
                                rnn.char_to_ix = char_to_ix
                                rnn.ix_to_char = ix_to_char
                                rnn.chars = chars            
                        
                                # BUILD THE MODEL
                                model = rnn.build()
                                # TRAIN THE MODEL
                                if not bigText:
                                    print("=====Text file not too big, do normal training=====")
                                    # only train if model has not been trained alread
                                    if alreadyProcessed(source,rnn.result_folder, rnn.result_path):
                                        print("This configuration has already been processed") 
                                        if (location == 0):
                                            call(["rm","-rf",rnn.result_folder])
                                        del rnn
                                        continue
                                    trained_model = rnn.train(model, X_train,y_train, X_val, y_val)
                                else:
                                    print("=====Text file too big for memory, use fit_generator for training=====")
                                    if alreadyProcessed(source,rnn.result_folder, rnn.result_path):
                                        print("This configuration has already been processed")
                                        if location == 0:
                                            call(["rm","-rf",rnn.result_folder])
                                        del rnn
                                        continue
                                    trained_model = rnn.train_generator(model, train_text, val_text)
                                # SAMPLE TEXT
                                sampleText = rnn.generate(trained_model, text)
                                with open(rnn.sample_path,"w") as f:
                                    f.write(sampleText)
                                ########################################
                                # add foldername to logfile
                                sourceFile = os.path.basename(source)
                                logfile_path = os.path.join(rnn.result_path, sourceFile.rstrip(".txt")+"_log.txt")
                                with open(logfile_path,"a") as f:
                                    f.write("\n")
                                    f.write(os.path.basename(rnn.result_folder))
                                # zip the folder (so that it can be transferred via ssh later)
                                """
                                if location == 0:
                                    call(["zip","-r",rnn.result_folder+".zip",rnn.result_folder])
                                    call(["rm","-rf",rnn.result_folder])
                                """
                                # delete created objects
                                del sampleText
                                del rnn
                    

        
if __name__ == "__main__":
    main()
########
#SSH COMMANDS
#
# scp na02zijy@faui00a.informatik.uni-erlangen.de:/home/cip/2014/na02zijy/Documents/WS1617/DL/Code/keras_char_prediction.py ./