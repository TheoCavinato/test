# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:45:15 2019

@author: THEO
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import csv


with open('C:/Users/THEO/Desktop/Biologie-IV/Programmation/data/sex.csv', 'r') as csvSex:
    reader = csv.reader(csvSex)
    dfSex = pd.DataFrame()
    for row in reader:
        dfSex = dfSex.append({'id':row[0], 'sex':row[1]}, ignore_index=True)
        
with open('C:/Users/THEO/Desktop/Biologie-IV/Programmation/data/colaus1.focus.raw.csv', 'r') as csvData:
    reader = csv.reader(csvData)
    dfData = pd.DataFrame()
    i=0
    for row in reader:
        dfData= dfData.append({'id':i, 'data':row}, ignore_index=True)
        i=i+1
        
dfFinal = dfData.join(dfSex, lsuffix='_caller', rsuffix='_other')

#si je veux utiliser le dataframe, il me faut en réalité créer deux tableaux de même taille : un contenant 
#les coordonnées, et un contenant les sexs
#il va faloir récupérer que les personnes dont le sex est spécifié -> ah non pas besoin, déjà résolu ça avec le df

ordonnees = dfFinal.loc[1:, 'data'].as_matrix()
j=0
while j < len(ordonnees):
    ordonnees[j] = [float(i) for i in ordonnees[j]]
    j=j+1
sex = dfFinal.loc[1:, 'sex'].as_matrix()
sex = [int(i) for i in sex]

#créer le réseau de neurones

model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])
    
#compiler
    
model.compile(optimizer='adam', 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

#entrainer, première variable les images, deuxième les labels (c'est ce qu'on veut lier aux images)

model.fit(ordonnees, sex, epochs=5)
