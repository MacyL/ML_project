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
## for output
import simplejson

####################
# Goal : create an corpus with some features.
####################
# Download some articles, here we download 10 articles each time. There are breast cancer, liver cancer and diabetes.
Entrez.email = "rorotira@gmail.com" 
handle=Entrez.esearch(db="pubmed",retmax=10, term="Diabetes [Title/Abstract] AND polymorphism AND genetics")
record=Entrez.read(handle)
handle.close()
list = record["IdList"]
out_handle=open("output4.txt","a")
for index in range(0, len(list)):
	listId = list[index]
	fetchHandle = Entrez.efetch(db="pubmed",rettype="MEDLINE", id=listId)
	data = fetchHandle.read()
	out_handle.write(data)

## after download, parse the MEDLINE format into three list : Title list (in word token), "Other term" list and abstract list (in sentence token)  
abstract=[]
title=[]
key=["OT","MH"]
otherT=[]
with open("output4.txt") as handle:
	records=Medline.parse(handle)
	for record in records:	
		ab=record['AB']
		ti=record['TI']
		abS=sent_tokenize(ab)
		abstract.extend(abS)
		tiw=word_tokenize(ti)
		title.extend(tiw)
		for i in key:
			if i in record.keys():
				data=record[i]
				otherT.extend(data)

####################################################################
# out file is train.txt, train2.txt, train3.txt, train4.txt
####################################################################
afile = open('train5.txt', 'w')
afile.write("\n".join(abstract))
afile.close()

####################################################################
# create a file with all the titles (word tokens)
####################################################################
title=list(set(title))
pre=['to','the','on','in','at','and','of','above','under','but','.','A','T','G','C','(',')',':']
pre2=['a','the','is','with']
title=[x for x in title if x not in pre]
title=[x for x in title if x not in pre2]
tfile=open('title.txt','w')
simplejson.dump(title, tfile)
tfile.close()
# when load, use simplejson.load(f)

####################################################################
# create a file with all the other term or MaSH term
####################################################################
otherT=list(set(otherT))
ofile=open('otherT.txt','w')
simplejson.dump(otherT, ofile)
ofile.close()


####################################################################
# The worth or not worth to put in final summary was manually judged. 
# csv file for abstract sentence created partically by manual 
###################################################################
