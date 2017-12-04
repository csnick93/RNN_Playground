# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 12:41:01 2016

@author: nickv
"""

source_path = r"C:\Users\nickv\Documents\WS1617\DL\Coding\CharacterPrediction\SourceText\RandomFile.txt"
log_file = r"C:\Users\nickv\Documents\WS1617\DL\data_generator_log.txt"

seq_length = 20

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
    
##########################################################################################

def data_generator(data, batch_size,return_target=True):
        '''
        This function is similar to gen_data. Yet, it produces batches of size |batch_size|
        infinitely -> will go over text repeatedly
        -> we always step one character forward with each succeeding generated sequence (so we produce overlap of seq_length-1)
        '''
        counter = 0
        while(True):
            if ((counter+1)*batch_size+seq_length < len(data)):
                start_char = counter*batch_size
                counter +=1 
            else:
                start_char = 0
                counter = 0
            
            x = []
            y = []
            for i in range(batch_size):
                x.append(data[start_char:start_char+seq_length])
                y.append(data[start_char+seq_length])
                start_char +=1
            with open(log_file,"a") as f:
                for i in range(len(x)):
                    f.write(x[i])
                    f.write("\t:\t"+ y[i])
                    f.write("\n")
                    f.write("===")
                f.write("\n\n\n")
                f.write("=====================================")
            yield x, y
            
    
text,train_text, val_text, vocab_size, char_to_ix, ix_to_char,chars = loadText(source_path,0.1)

for i,(x,y) in enumerate(data_generator(text,10)):
    if( i > 10000):
        break
    