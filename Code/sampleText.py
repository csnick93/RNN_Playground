# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 15:50:46 2016

@author: nickv

Sample text from models
"""




import os
from keras.models import load_model
import numpy as np

result_path = r"/proj/ciptmp/na02zijy/Results/bigTextfile"
text_path = r"/proj/ciptmp/na02zijy/Results/bigTextfile.txt"

#################################################################################
"""
compute test loss for all models
"""

def evaluateModelsOnTestSet(result_path, testSetPath,seq_len):
    testText = open(testSetPath).read()
    for d in os.listdir(result_path):
        if not os.path.exists(os.path.join(result_path,d,"history.txt")):
            continue
        testLoss = computeTestloss(os.path.join(result_path,d), testText, seq_len)
        with open(os.path.join(result_path,d,"history.txt"),'a') as f:
            f.write("\nTest Loss: "+str(testLoss))
#################################################################################
def computeTestloss(folder_path, testText, seq_len):
    log_file = os.path.join(folder_path,"test_log.txt")
    if os.path.exists(log_file):
        os.remove(log_file)
    print("=== Computing test loss for : "+ folder_path + "===" )
    # load model
    model = load_model(os.path.join(folder_path,"model.h5"))
    # load text that we are testing on
    # get text data that we trained on
    text,train_text, val_text, vocab_size, char_to_ix, ix_to_char,chars = loadText(text_path, .1)
    # score
    score_top1 = 0
    score_top5 = 0
    # iterate over test text: predict next character and check if its correct
    for i in range(len(testText)-seq_len):
        sequence = testText[i:seq_len+i]
        correct_char = testText[seq_len+i]
        x = np.zeros((1,seq_len,vocab_size))
        x[0] = sequence_encoder(sequence, vocab_size, char_to_ix)
        # this produces a probability distribution over characters
        probs = model.predict(x, verbose=0)[0]
        char_top1 = ix_to_char[np.argpartition(probs,-1)[-1:][0]]
        char_top5 = [ix_to_char[c] for c in np.argpartition(probs,-5)[-5:]]
        # sample the character to use based on the predicted probabilities (use temperature = 0 -> argMax of probabilities)
        #next_idx = sample(probs, temperature = 0.1)
        #next_char = ix_to_char[next_idx]
        if char_top1 == correct_char:
            score_top1 +=1
        if correct_char in char_top5:
            score_top5 +=1
        print(i)
        """
        with open(log_file,"a") as f:
            f.write("Sequence: " + sequence)
            f.write("\nPredicted Char: " + next_char)
            f.write("\nCorrect Char: " + correct_char)
            f.write("\n==============")
        """
    return (np.round(score_top1*1./(len(testText)-seq_len),2),np.round(score_top5*1./(len(testText)-seq_len),2)) 
#################################################################################
"""
sample text (let network dream) for all trained models
"""
def sampleForAllModels(result_path, seed, predict_length, temperature):
    for d in os.listdir(result_path):
        if not os.path.exists(os.path.join(result_path,d,"history.txt")):
            continue
        #try:
        sampledText = sampleText(os.path.join(result_path,d),seed,predict_length, temperature)
        #except Exception:
        #    print("No sampling possible for "+ d)
        #    continue
        with open(os.path.join(result_path,d,"mySampleText.txt"),"w") as f:
            f.write(sampledText)



##################################################################################
"""
folderPath: path to a result folder, containing history.txt
        /home/cip/2014/na02zijy/Documents/WS1617/DL/Results/bigTextfile/Folder_1_10_20_GRU_False_0.2_False_100_512        
"""
def sampleText(folderPath,seed, predict_length, temperature):
    print("===Sampling text for "+folderPath + "===")
    source = folderPath.split("/")[-2]
    if not os.path.exists(os.path.join(folderPath,"model.h5")):
        print("No model available at : "+ folderPath)
        return
    model = load_model(os.path.join(folderPath,"model.h5"))
    if source == "bigTextfile":
        text,train_text, val_text, vocab_size, char_to_ix, ix_to_char,chars = loadText(text_path, 0.1)
        #print(vocab_size)
    elif source == "accumulatedCode":
        text,train_text, val_text, vocab_size, char_to_ix, ix_to_char,chars = loadText(text_path,.1)
    sampleText = generate(model, seed, predict_length, temperature, vocab_size, ix_to_char, char_to_ix,seq_length = 20)
    return sampleText

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
    
    return text,train_text, val_text, vocab_size, char_to_ix, ix_to_char,chars
    
    

###############################################################################
# SAMPLE TEXT #
###############
    
"""
model: trained model
temperature: randomness of sampling
seed: initial sequence of characters
seq_length: how many characters are considered for prediction
predicate: determines when we are done predicting 
"""    
    
def generate(model, seed, predict_length, temperature, vocab_size, ix_to_char, char_to_ix,seq_length = 20):
    print("===GENERATING SAMPLE TEXT===")
    predicate=lambda x: len(x) < predict_length
    # if seed does not meet requirements, randomly choose chunk of text from f
    if len(seed) < seq_length:
       print("Seed is not long enough, cannot start sampling text!")
       return
    
    sentence = seed[:seq_length]
    generated = seed        # here our predicted string is being generated

    while predicate(generated):
        # generate the input tensor
        # from the last max_len characters generated so far
        # encode the input sequence
        x = np.zeros((1,seq_length,vocab_size))
        x[0] = sequence_encoder(sentence[-seq_length:], vocab_size, char_to_ix)

        # this produces a probability distribution over characters
        probs = model.predict(x, verbose=0)[0]

        # sample the character to use based on the predicted probabilities
        next_idx = sample(probs, temperature)
        next_char = ix_to_char[next_idx]

        generated += next_char
        sentence = sentence[1:] + next_char
    return generated

###############################################################################
#  DATA ENCODER   #
###################
# encode a sequence of chars into 0-1 vectors
def sequence_encoder(data, vocab_size, char_to_ix):    
    x = np.zeros((len(data),vocab_size))    # encode chars in sequence for the |batch-size| sequences
    # encode |batch_size| sequences of length SEQ_LENGTH (for each new sequence, move STEP many characters ahead (produce overlap))
    for i in range(0, len(data)):
        x[i,char_to_ix[data[i]]] = 1.
    return x

################################################################################

def sample(probs, temperature):
    # the higher the temperature, the more likely an unlikely character will be chosen
    """samples an index from a vector of probabilities"""
    a = np.log(probs)/temperature
    a = np.exp(a)/np.sum(np.exp(a))
    a = a - ((np.sum(a)-1)/len(a)) - 1e-8         # so that we have a sum of slightly below one in array to represent a pdf (impossible to have exactly sum of 1 due to numerical instabilities, therefore try to get slightly below)
    return np.argmax(np.random.multinomial(1, a, 1))
    
###############################################################################

if __name__ == "__main__":  
    seed = "Sherlock is really a"
    predict_length = 500
    temperature = .35
    sampleForAllModels(result_path,seed,predict_length,temperature)
    #
    evaluateModelsOnTestSet(result_path, r"/home/cip/2014/na02zijy/Documents/WS1617/DL/SourceText/SherlockTestSet.txt",20)
    
    