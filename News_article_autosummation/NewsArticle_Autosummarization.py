# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 15:01:56 2018

@author: 1rahu
"""

import nltk
nltk.download('stopwords')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest 

class FrequencySumarizer:
    def __int__(self, min_cut=.1, max_cut=.9):
        self._min_cut=min_cut
        self._max_cut=max_cut
        self._stopwords=set(stopwords.words('english') + list(punctuation))
        
    def _count_frequency(self, word_sent):
        """
        this function will take list of sentences in the texxt and returns the
        dictionary of words and there respective frequencies .
        """
        freq=defaultdict(int)
        
        for sent in word_sent:
            for word in sent:
                if word not in self._stopwords:
                  freq[sent]+=1;
    
        """
        we will normalize each frequency by dividing each freq with the maximum frequency
        and keeping only those frequency wihich lies between the min_cut and max_cut value
        """
    
        max_freq=float(max(freq.values()))
        for word in freq.keys():
           freq[word]/=max_freq
           if freq[word] >= self._max_cut or freq[word] <= self._min_cut:
             del(freq[word])
        return freq
                
    def summarize(self, text, n):
        sent=sent_tokenize(text)
        
        assert n <= len(sent)
        
        word_sent = [word_tokenize(s.lower()) for s in sent]
        
        self._freq = self._count_frequency(word_sent)
        ranking = defaultdict(int)
        
        for i,sent in enumerate(word):
            for word in sent:
                ranking[i]+=self._freq[word]
        sent_indx = nlargest(n, ranking, key=ranking.get)
        return (sent[j] for j in sent_indx)
            
    
    
import urllib.request
from bs4 import BeautifulSoup

def get_text_url(url):
    page = urllib.request.urlopen(url).read().decode('utf8')
    soup=BeautifulSoup( page )
    text = ''.join(map(lambda p: p.text, soup.find_all('article')))
    
    soup2=BeautifulSoup( text )
    text=''.join(map(lambda p: p.text, soup2.find_all('p')))
    return soup.title.text , text

someurl = "https://www.washingtonpost.com/news/the-switch/wp/2015/08/06/why-kids-are-meeting-more-strangers-online-than-ever-before/"
textofurl = get_text_url(someurl)

fs = FrequencySummarizer()
summary = fs.summarize(textOfUrl[1], 3)
    
    

        
        
        
