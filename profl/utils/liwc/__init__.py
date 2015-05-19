#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
A module that allows for the creation of frequency dictionaries (relative or absolute)
for the LIWC categories in either Dutch or English.
"""

# ---------------------------------------------------------------------------------

#  Attention! 
#  This program may only be used if the user has permission to use the LIWC data.
#  For information on the Linguistic Inquiry and Word Count, visit http://www.liwc.net
#
#  This program uses the Dutch 2001 version and the English 2007 version of LIWC.

# ---------------------------------------------------------------------------------

from __future__ import division
from __future__ import print_function
import os
from codecs import open

__date__ = '5 May 2014'
__version__ = '1.2'
__author__ = 'Ben Verhoeven'


#------------ DUTCH DATA --------------


this_dir, this_file = os.path.split(__file__)
csv = os.path.join(this_dir, "LIWC_Dutch.csv")		#load Dutch LIWC data

csvfile = open(csv,"r", encoding='utf-8')
liwcfile = csvfile.read().split("\n")
csvfile.close()

liwc_nl_dict = dict()
for line in liwcfile:
	line = line.rsplit(",")
	liwc_nl_dict[line[0]] = line[1:]
		
	
#----------- ENGLISH DATA -------------


csv = os.path.join(this_dir, "LIWC_English.csv")	#load English LIWC data

csvfile = open(csv,"r",encoding='utf-8')
liwcfile = csvfile.read().split("\n")
csvfile.close()

liwc_en_dict = dict()
for line in liwcfile:
	line = line.rsplit(",")
	liwc_en_dict[line[0]] = line[1:]


#----------- FUNCTIONS ----------------


def freqdict(text):

	"""This function returns a frequency dictionary of the input list. All words are transformed to lower case."""
	
	freq_dict = dict() 
	for word in text:
		word = word.lower()
		if word in freq_dict:
			freq_dict[word] += 1
		else:
			freq_dict[word] = 1
	return freq_dict


def liwc(text,output='rel',lang='en'):

	"""This function takes a list of tokens as input and returns a dictionary with the relative (output='rel') or absolute (output='abs') frequencies for every LIWC category. This function works for languages English (lang='en') and Dutch (lang='nl')."""

	#if text == string: convert string to list
	if type(text).__name__ in ('str','unicode'):
		text = text.split(" ")
		print("\nATTENTION: Your input was of type 'string' or 'unicode' instead of 'list'. To be able to process it, we have split this string at the spaces. This may have an effect on the output, since your text has not been properly tokenized.\n\n")
	
	#decide on relative or absolute frequenc
	if output == 'abs': #absolute frequency as output
		division = 1
	elif output == 'rel': #relative frequency as output
		division = len(text)

	#make frequency dictionary of the text to diminish number of runs in further for loop
	freq_dict = freqdict(text) 	
	
	if lang == 'nl':
		liwc_dict = liwc_nl_dict
	else:
		liwc_dict = liwc_en_dict
	
	features = dict()		
	for category in liwc_dict:
		freq = 0
		for term in liwc_dict[category]:
			term = term.lower()
			if term[-1] == u"*": #'*' indicates partial words that should match the beginning of the word (include variations on words)
				for word in freq_dict:
					if word.startswith(term[:-1]):
						freq += freq_dict[word]
			else:
				if term in freq_dict:
					freq += freq_dict[term]
		features[category] = freq / division
		
	return features

def liwc_nl(text,output="rel"):
	"""This function applies Dutch liwc() on input. Output is relative frequencies of liwc categories."""
	return liwc(text,output=output,lang="nl")

def liwc_en(text,output="rel"):
	"""This function applies English liwc() on input. Output is relative frequencies of liwc categories."""
	return liwc(text,output=output,lang='en')
	
#END	