# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 22:24:19 2016

@author: nickv

GUI to create html report summary.
"""

import os
from tkinter import *
from htmlReport import makeHTMLReport
from helper_functions import collect_options

class Checkbar(Frame):
    def __init__(self, parent=None, picks=[], side=LEFT, anchor=W):
        Frame.__init__(self, parent)
        self.vars = []
        for pick in picks:
            var = IntVar()
            var.set(1)
            chk = Checkbutton(self, text=pick, variable=var)
            chk.pack(side=side, anchor=anchor, expand=YES)
            self.vars.append(var)
    def state(self):
        return map((lambda var: var.get()), self.vars)

def main():
    root = Tk()
    sourceTextOptions, unitTypeOptions, layerOptions, epochOptions =  \
        collect_options(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'Results'))
    layerOptions = [l+' layer(s)' for l in layerOptions]
    epochOptions = [e + ' epochs' for e in epochOptions]
    
   
    sourceText = Checkbar(root,sourceTextOptions)
    unitType = Checkbar(root,unitTypeOptions)
    epochs = Checkbar(root, epochOptions)
    numberLayers = Checkbar(root, layerOptions)
    sourceText.pack(side=TOP,  fill=X)
    unitType.pack(side=TOP,  fill=X)
    epochs.pack(side = TOP,fill=X)
    numberLayers.pack(side = TOP,fill = X)
    def callWebReportCreation(): 
        '''
        Get checked options and create HTML report.
        '''
        chosenSources = [sourceTextOptions[i] for i in [j for j,k in enumerate(list(sourceText.state())) if k==1]]
        chosenUnits = [unitTypeOptions[i] for i in [j for j,k in enumerate(list(unitType.state())) if k==1]]
        chosenEpochs = [epochOptions[i] for i in [j for j,k in enumerate(list(epochs.state())) if k==1]]
        chosenLayers = [layerOptions[i] for i in [j for j,k in enumerate(list(numberLayers.state())) if k==1]]
        result_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'Results')
        html_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'html_report.html')
        makeHTMLReport(html_file, result_folder, chosenSources,chosenUnits,chosenEpochs,chosenLayers)
    Button(root, text='Create Summary', command=callWebReportCreation).pack(side=RIGHT)
    root.mainloop()


if __name__ == '__main__':
   main()
