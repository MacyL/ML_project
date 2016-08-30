###########
# Library
###########
## Biopython for searching PubMed
from Bio import Entrez
from Bio import Medline
# text process 
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import re
import pandas as pd
import simplejson
# for model
import theano
import numpy as np
from collections import Counter
from sklearn.cross_validation import train_test_split,KFold
from sklearn.cross_validation import cross_val_score
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding,Input, Dense,Dropout, Activation, Flatten,Convolution1D, MaxPooling1D
from keras.layers import Reshape, Convolution2D, MaxPooling2D
from keras.models import Sequential,Model
from keras.preprocessing import sequence
import keras.preprocessing.sequence as ks
from keras.preprocessing.sequence import pad_sequences
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sknn.mlp import Classifier, Layer
from keras.models import Sequential,Model

##################################################
# Goal: build  a model and train on the small data
###################################################

# input abstract sentence and y score (worth to put in summary or not) 
inp=pd.read_csv('train.csv',sep='\t')
t=open('title.txt','r')
title=simplejson.load(t)
o=open('otherT.txt','r')
otherT=simplejson.load(o)
ab=inp['Sentence']
Y=inp['Worth']
############################## split into training data and testing data ####################
X_train, X_test, y_train, y_test = train_test_split(ab, Y, test_size=0.2, random_state=42)

# function for calculating the features 
def words_in_string(word_list, a_string):
    return set(word_list).intersection(a_string.split())

# feature 1= if the sentence in "abstract" list contains any words in title 
collection_t=dict()
for i in X_train:
	t=words_in_string(title,i)
	collection_t[i]=len(t)/len(i)

# feature 2=if the sentence in "abstract" list contains any words in otherT
collection_ot=dict()
for i in X_train:
	t=words_in_string(otherT, i)
	collection_ot[i]=len(t)/len(i)


# feature 3= sentence to sequence, maximum lenth is 30 words. 

tokenizer = Tokenizer(nb_words=30)
tokenizer.fit_on_texts(X_train)
sequences = tokenizer.texts_to_sequences(X_train)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=30)

# merge these features? 
title_count=pd.DataFrame.from_dict(collection_t,orient='index')
otherT_count=pd.DataFrame.from_dict(collection_ot,orient='index')
theMerge=pd.concat([title_count, otherT_count],axis=1)
theMerge_matrix=theMerge.as_matrix()
features=dict()
for i in range(len(theMerge_matrix)):
	temp=np.concatenate((data[i],theMerge_matrix[i]))
	features[i]=temp

features_count=pd.DataFrame.from_dict(features, orient='index')
features_matrix=features_count.as_matrix()

model=MLP.fit(features_matrix,y_train)

# test data 
collection_t=dict()
for i in X_test:
	t=words_in_string(title,i)
	collection_t[i]=len(t)/len(i)

collection_ot=dict()
for i in X_test:
	t=words_in_string(otherT, i)
	collection_ot[i]=len(t)/len(i)

tsequences = tokenizer.texts_to_sequences(X_test)
word_index = tokenizer.word_index
tdata = pad_sequences(tsequences, maxlen=30)

title_count=pd.DataFrame.from_dict(collection_t,orient='index')
otherT_count=pd.DataFrame.from_dict(collection_ot,orient='index')
theMerge=pd.concat([title_count, otherT_count],axis=1)
theMerge_matrix=theMerge.as_matrix()
tfeatures=dict()
for i in range(len(theMerge_matrix)):
	temp=np.concatenate((tdata[i],theMerge_matrix[i]))
	tfeatures[i]=temp

tfeatures_count=pd.DataFrame.from_dict(tfeatures, orient='index')
tfeatures_matrix=tfeatures_count.as_matrix()

MLP = Classifier(
    layers=[
        Layer("Sigmoid", units=150),
        Layer("Softmax")],
    learning_rate=0.02,
    n_iter=10)

MLP.fit

predict_result=MLP.predict(tfeatures_matrix)
y_matrix=y_test.as_matrix()
tout=pd.DataFrame(predict_result, X_test)



# save for later  
#features_count=np.array(list(features.values()))
##################################################above is official #####################################################################
################################################## below is not official
# try MLP first. we repeat the above work but with full data set, sklearn can split data into train and test set 10 folds . 

# feature 1= if the sentence in "abstract" list contains any words in title 
collection_t=dict()
for i in ab:
	t=words_in_string(title,i)
	collection_t[i]=len(t)/len(i)

# feature 2=if the sentence in "abstract" list contains any words in otherT
collection_ot=dict()
for i in ab:
	t=words_in_string(otherT, i)
	collection_ot[i]=len(t)/len(i)


# feature 3= sentence to sequence, maximum lenth is 30 words. 

tokenizer = Tokenizer(nb_words=40)
tokenizer.fit_on_texts(ab)
sequences = tokenizer.texts_to_sequences(ab)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=40)

# merge these features? 
title_count=pd.DataFrame.from_dict(collection_t,orient='index')
otherT_count=pd.DataFrame.from_dict(collection_ot,orient='index')
theMerge=pd.concat([title_count, otherT_count],axis=1)
theMerge_matrix=theMerge.as_matrix()
features=dict()
for i in range(len(theMerge_matrix)):
	temp=np.concatenate((data[i],theMerge_matrix[i]))
	features[i]=temp

features_count=pd.DataFrame.from_dict(features, orient='index')
features_matrix=features_count.as_matrix()

# MLP
MLP = Classifier(
    layers=[
        Layer("Sigmoid", units=200),
        Layer("Softmax")],
    learning_rate=0.02,
    n_iter=10)
LP_accuracy = cross_val_score(MLP, features_matrix, Y, cv=10, scoring='accuracy')
LP_accuracy
array([ 0.68965517,  0.71428571,  0.71428571,  0.64285714,  0.64285714,
        0.60714286,  0.67857143,  0.57142857,  0.57142857,  0.5       ])
# average=0.63

# strategy 2 : CNN

# CNN requires y in a matrix format 
y = np.zeros((len(features_matrix), 1))

for i in range(len(Y)):
	if Y[i] ==1:
		y[i]=[True] # positive
	else:
		y[i]=[False] # negative


scores_conv = []
kf_total = KFold(len(features_matrix), n_folds=10, shuffle=True, random_state=3)
for train_index, test_index in kf_total:
	myTrain=features_matrix[train_index]
	myTrainResponse=y[train_index]
	myTest=features_matrix[test_index]
	expected=y[test_index]
	model = Sequential()
	model.add(Embedding(42,32,input_length=42, dropout=0.2))
	model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='valid',activation='relu', subsample_length=1))
	model.add(MaxPooling1D(pool_length=model.output_shape[1]))
	model.add(Flatten())
	model.add(Dense(20))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(myTrain, myTrainResponse,batch_size=50, nb_epoch=10)
	score = model.evaluate(myTest, expected)
	scores_conv.append(score[1])

#average=0.67

