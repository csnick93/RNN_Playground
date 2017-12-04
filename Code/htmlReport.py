# -*- coding: utf-8 -*-
"""
Created on Fri Jul 01 12:53:57 2016

Create the webpage

@author: z003caje
"""
import os
import numpy as np
import webbrowser

##############################################################
def createReport(modelList,html_file):
    '''
    Method in charge of creating the html report.
    
    Args:
        modelList: list of paths of trained models to display
        html_file: file to write html code to
    '''
    createHTML_header(html_file, modelList)        
    for index,model in enumerate(modelList):
        createInstance(index,model,html_file)        
    createHTML_endOfFile(html_file)
    
##############################################################
def createHTML_header(html_file, modelList):
    """
    Create header html report.
    
    Args:
        html_file: where to write html code to
        modelList: list of network folders to put results on html
    """
    with open(html_file,'w') as f:
        f.write("<!DOCTYPE html>\n")
        f.write("<html>\n")
        f.write("<head>\n")
        f.write("\t<title> RNN Report </title>\n")
        f.write("\t<meta charset=\"utf-8\">\n")
        f.write("\t<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
        f.write("\t    <link rel=\"stylesheet\" href=\"http://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css\">\n")
        f.write("\t    <script type=\"text/css\">\n")
        f.write("\t\t.center{\n")
        f.write("\t\tmargin: 0 auto; width: 200px;\n")
        f.write("\t\t}\n")
        f.write("\t</script>\n")
        f.write("\t<style>\n")
        f.write("\t#tableData {\n")
        f.write("\t\tbackground-color: lightgrey;\n")
        f.write("\t\theight = 200px;\n")
        f.write("\t\twidth = 400px;\n")
        f.write("\t\tposition: relative;\n")#
        f.write("\t\ttop:65px;\n")#
        f.write("\t\tfloat:center;\n")
        f.write("\t\tpadding: 10px;\n")
        f.write("\t}\n")
        f.write("\ttable, th, td {\n")
        f.write("\t\tborder: 1px solid black;\n")
        f.write("\t\tborder-collapse: collapse;\n")
        f.write("}\n")
        f.write("\tth, td {")    
        f.write("\t\tpadding: 5px;\n")
        f.write("\t}\n")
        f.write("\t</style>\n")
        f.write("\t<script src=\"https://ajax.googleapis.com/ajax/libs/jquery/1.12.4/jquery.min.js\"></script>\n")
        f.write("\t<script src=\"http://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js\"></script>\n")
        f.write("</head>\n")
        f.write("\n")
        f.write("<body style = \"background-color:lightgrey;\">\n")
        f.write("<h1 id = \"TOP\" style=\"text-align:center;\"> RNN Report </h1>\n")
        f.write("<br>")
        f.write("<br>")
        f.write("<p> Number of trained Networks: "+ str(len(modelList))+"</p>\n")
        f.write("<br>")
        f.write("<br>")
        f.write("<br>\n")
        f.write("<div class=\"container-fluid\">\n")

#############################################################################
def createHTML_endOfFile(html_file):
    with open(html_file,'a') as f:
        f.write("</div>\n")
        f.write("</body>\n")
        f.write("</html>")
############################################################################

def createInstance(index,model_path,html_file):  
    '''
    Creating html instance for certain model.
    
    Args:
        index: the index for number of models in html report
        model_path: path to model
        html_file: html report file
    '''
    print("Creating: " + model_path)
    with open(html_file,'a') as f:
        ####################
        # CREATE THE TABLE #
        ####################
        with open(os.path.join(model_path,"history.txt"),'r') as h:
            hist = h.read()
            duration = np.int(np.float(hist.split("Training took: ")[1].split(" seconds")[0]))
            duration = str(int(duration/3600))+" h " + str(int((duration%3600)/60)) + " m "+ str(int(duration%3600%60))
            train_loss = hist.split("Training loss:")[1].split("Validation loss:")[0]
            val_loss = hist.split("Validation loss:")[1]
            train_loss = train_loss.lstrip("\n[").rstrip("]\n")
            val_loss = val_loss.lstrip("\n[").rstrip("]\n")
        #####
        train_loss = np.fromstring(train_loss, sep = " ")
        val_loss = np.fromstring(val_loss, sep = " ")
        best_val_loss = np.around(np.min(val_loss),2)
        best_train_loss = np.around(np.min(train_loss),2)
        header = ["Source Text", "Unit Type", "Number Of Layers", "Epochs", "Sequence Length", "Dropout Rate",
                  "HiddenSize", "Inner Dropout","L2-Regularization","Duration",
                  "Best Training Loss", "Best Validation Loss"]
        tableData = get_table_data(os.path.join(model_path,'model_config.txt'))
        tableData = tableData + [str(duration),str(best_train_loss),str(best_val_loss)]   
        f.write("\t <div class=\"row\">\n")
        f.write("\t\t<h2> Model %i </h2>\n"%index)
        f.write("\t\t <br> \n \t\t<br>\n")
        f.write("\t\t <div class = \"col-md-2\">\n")
        f.write("\t\t\t <h3 style=\"text-align:center;\">Parameter Table</h3>\n")
        f.write("\t\t\t <div id=\"tableData\">\n")
        f.write("\t\t\t\t <table style = \"width:100%\"\n")
        for i in range(len(header)):
            if header[i] == "Duration":
                f.write("\t\t\t\t <tr>\n")
                f.write("\t\t\t\t\t <td>"+ header[i]+"</td>\n")
                f.write("\t\t\t\t\t <td>"+ tableData[i]+" s</td>\n")
                f.write("\t\t\t\t </tr>\n")
            else:
                f.write("\t\t\t\t <tr>\n")
                f.write("\t\t\t\t\t <td>"+ header[i]+"</td>\n")
                f.write("\t\t\t\t\t <td>"+ tableData[i]+"</td>\n")
                f.write("\t\t\t\t </tr>\n")
        f.write("\t\t\t\t </table>\n")
        f.write("\t\t\t </div>\n")
        
        f.write("\t\t </div>\n")
        f.write("\n")
        
        #############
        # LOSS PLOT #
        ############
        f.write("\t\t <div class=\"col-md-5\">\n")
        f.write("\t\t\t<h3 style=\"text-align:center;\">Loss Graph</h3>\n")
        
        f.write("<br>\n<br>\n<br>\n<br>\n<br>\n")
        loss_plot_path = './Results/'+os.path.basename(model_path)+'/loss_plot.png'
        f.write("<img src=\""+loss_plot_path+ "\" alt = \"Loss Graph\" style=\"display:block; margin: auto;\" width=\"450\" height=\"325\">")
        
        f.write("\t\t</div>\n")
        f.write("\n")
        
        #####################
        # WRITE SAMPLE TEXT #
        #####################
        
        f.write("\t\t <div class=\"col-xs-12 col-md-5\">\n")
        f.write("\t\t\t<h3 style=\"text-align:center;\">Sample Text</h3>\n")
        
        f.write("<br>\n")
        f.write("<br>\n")
        f.write("<br>\n")
        textFormatted = formatText(os.path.join(model_path,"sampleText.txt"))
        for line in textFormatted:
            f.write("\t\t\t\t <p align=\"center\"> "+line+"</p>\n")
        f.write("\t\t</div>\n")
        
        f.write("\t</div>\n")
        f.write("\n")

        f.write("<br>\n<br>\n<br>\n<br>\n<br>\n")
        f.write("<p><a href=\"#TOP\">Back to the Top</a></p>")
        f.write("<br>\n<br>\n<br>\n<br>\n<br>\n")


###############################################################################
def get_table_data(config_file):
    '''
    Helper file to get required table data from config file.
    
    Args:
        config_file: path to configuration file
        
    Returns:
        Required table data in correct order
    '''
    table_data = []
    with open(config_file,'r') as config:
        config_txt = config.read()
        table_data.append(os.path.basename(config_txt.split('source_path: ')[1].split('\n')[0]))
        table_data.append(config_txt.split('unit_type: ')[1].split('\n')[0])
        table_data.append(config_txt.split('no_layers: ')[1].split('\n')[0])
        table_data.append(config_txt.split('epochs: ')[1].split('\n')[0])
        table_data.append(config_txt.split('seq_length: ')[1].split('\n')[0])
        table_data.append(config_txt.split('dropout: ')[1].split('\n')[0])
        table_data.append(config_txt.split('hidden_size: ')[1].split('\n')[0])
        table_data.append(config_txt.split('inner_dropout: ')[1].split('\n')[0])
        table_data.append(config_txt.split('l2_regularization: ')[1].split('\n')[0])
    return table_data          
    
############################################################################### 

def formatText(text_path):
    """
    Format text by splitting at spaces and equally distributing the words.
    
    Args:
        text_path: path to the sampled text file
    
    Returns:
        Formatted text.
    """    
    f = open(text_path)
    lineLength = 60                
    text = f.read()
    textFormatted = []
    textCounter = 0
    line = ""
    while textCounter < len(text):
        # check if we can do line break
        if(len(line) >= lineLength and text[textCounter] == " " ):
            #line += text[textCounter]
            textFormatted.append(line)
            line = ""
            textCounter +=1
        elif(textCounter +1 == len(text)):
            line += text[textCounter]
            textFormatted.append(line)
            textCounter +=1
        else:
            line += text[textCounter]
            textCounter+=1
    return textFormatted
###############################################################################

def fitsCriteria(model_config,source,units,epochs,numberOfLayers):
    """
    Check if a folder fits the specified criteria for being shown.
    
    Args:
        model_config: path to config file of model
        source: source options clicked in GUI
        units: unit options clicked in GUI
        epochs: epoch options clicked in GUI
        numberOfLayers: layer options clicked in GUI
        
    Returns:
        Flag indicating whether model should be displayed
    """
    source_text, unit_type, no_layers, epochs_training = get_relevant_data(model_config)
        
    if (source_text in source and unit_type in units and epochs_training in epochs 
            and no_layers in numberOfLayers):
        return True
    else:
        return False
    
###############################################################################
def get_relevant_data(config_file):
    '''
    Small helper that extracts:
        unit_type, source_text, epochs, no_layers
        
    Args:
        config_file: path to configuration file
        
    Returns:
        unit_type, source_text, epochs, number_layers
    '''
    with open(config_file,'r') as config:
        config_txt = config.read()
        unit_type = config_txt.split('unit_type: ')[1].split('\n')[0]
        source_text = os.path.basename(config_txt.split('source_path: ')[1].split('\n')[0])
        no_layers = config_txt.split('no_layers: ')[1].split('\n')[0]
        epochs_training = config_txt.split('epochs: ')[1].split('\n')[0]
        
    return source_text, unit_type, no_layers, epochs_training
        
###############################################################################

def sortList(model_folders):
    """
    Sort the networkList according to:
        1) Source
        2) Epoch Length
        3) Number of Layers
        4) Unit Type (SimpleRNN, LSTM, GRU)
        
    Args:
        model_folders: list of paths to model folders
    """
    sortedList = [model_folders[0]]
    for i in range(1,len(model_folders)):
        j = 0
        while(j < len(sortedList) and isGreaterThan(model_folders[i],sortedList[j])):
            j += 1
        sortedList.insert(j, model_folders[i])
    return sortedList

###############################################################################

def isGreaterThan(path, comparePath):
    """
    Compares two model folders which one to display first.
    
    Args:
        path: reference model folder
        comparePath: to be compared model folder
    Return: 
        True if path > comparePath, False otherwise
    """
    source_text, unit_type, no_layers, epochs_training = \
        get_relevant_data(os.path.join(path,'model_config.txt'))
    source_text_, unit_type_, no_layers_, epochs_training_ = \
        get_relevant_data(os.path.join(comparePath,'model_config.txt'))

    if source_text > source_text_:
        return True
    elif source_text < source_text_:
        return False
    else:
        if epochs_training > epochs_training_:
            return True
        elif epochs_training < epochs_training_:
            return False
        else:
            if no_layers > no_layers_:
                return True
            elif no_layers < no_layers_:
                return False
            else:
                if unit_type_ == "GRU":
                    return True
                elif no_layers == "LSTM" and unit_type_ != "GRU":
                    return True
                else:
                    return False
###############################################################################
 
def makeHTMLReport(outFile, result_folder, sourceFile, unitType, epochs, numberOfLayers):
    """
    Create webReport with networks fitting to specified requirements:
        sourceFile, unitType, epochs, numberOfLayers
        
    
    Args:
        outfile: path to html file being written to
        result_folder: root folder where model result folders lie
        sourceFile: sourceFile options
        unitType: the RNN unit options
        epochs: the epoch options
        numberOfLayers: the different layer configurations
    """   
    print("Creating WebReport...")
    epochs = [e.rstrip(" epochs") for e in epochs]
    numberOfLayers = [l.rstrip(" layer(s)") for l in numberOfLayers]
    networkList = []
    # collect all trained networks that fit criteria (paths of respective result folders)
    for folder in os.listdir(result_folder):
        if fitsCriteria(os.path.join(result_folder,folder,'model_config.txt'),
                        sourceFile, unitType, epochs, numberOfLayers):
            networkList.append(os.path.join(result_folder,folder))
        
    networkList = sortList(networkList)
    createReport(networkList,outFile)
    webbrowser.open(outFile)
    print("Done")
    
############################################################################### 
if __name__ == "__main__":
    pass