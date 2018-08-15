# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 19:56:14 2018

@author: 1rahu
"""

import twitter

api= twitter.api( 
        
        )
print(api.VerifyCredentials())

def createTestData(search_string):
    try:
        tweet_fetched=api.GetSearch(search_string, count=100)
        print str(len(tweet_fetched))+" has been succesfully Fetched with the search term "+search_string
        # We will fetch only the text for each of the tweets, and since these don't have labels yet, 
        # we will keep the label empty 
        return [{"text":status.text, "label":None} for status in tweet_fetched]
    except:
        print " Error occured "
        return None

search_string=input(" Enter the tech related string to be searched ")
testData = createTestData(search_string)

testData[0:9]

def createTrainingCorpus(corpusFile,tweetDataFile):
    import csv
    corpus[]
    with open(corpusFile, rb) as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',',quotechar="\"")
        for row in lineReader:
            corpus.append({"tweet_id":row[2], "Label ": row[1], "Topic":row[0]})
    
    import time
    rate_limit = 180
    sleep_time = 900/180
    
    trainingData=[]
    for tweet in corpus:
        try:
            status = api.GetStatus(tweet["tweet_id"])
            print "Tweet_fetched is : "+ status.text
            tweet["text"] = status.text
            trainingData.append(tweet)
            time.sleep(sleep_time)
        except:
            continue
        
        with open(tweetDataFile,'wb') as csvfile:
            linewriter=csv.writer(csvfile,delimiter=',',quotechar="\"")
            for tweet in trainingData:
                try:
                    linewriter.writerow([tweet["tweet_id"],tweet["text"],tweet["label"],tweet["topic"]])
                except Exception, e:
                    print e
        return trainingData
                    
                
## setting up of createtrainingCorpus for corpusFile and tweetDataFile, after 
## installing all the required file.

#### trainingData=createLimitedTrainingCorpus(corpusFile,tweetDataFile)
import re
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords

class PreProcessTweets:
    def __init__(self):
        self._stopwords=set(stopwords.words('english')+ list(punctuation) + ['AT_USER', 'URL'])
        
    def processTweets(self, list_of_tweets):
        processedTweets=[]
        
        for tweet in lis_of_tweets:
            processedTweet.append(self._processTweets(tweet["text"]), tweet["label"])
    return processedTweets
    

    def _processTweets(self, tweet):
        tweet=tweet.lower()
        tweet=re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL' , tweet )
        tweet=re.sub('@[^\s]+','AT_USER',tweet)
        tweet=re.sub(r'#([^\s]+)',r'\1',tweet)
                     
        tweet=word_tokenize(tweet)
        return [word for word in tweet if word not in self._stopwords]
    
tweetProcessor=PreProcessTweets()
ppTrainingData=tweetProcessor.processTweets(trainingData)
ppTestData=tweetProcessor.processTweets(testData)  

import nltk

def bulidVocabulary(ppTrainingData):
    all_words[]
    for (words, sentintiments) in ppTrainingData:
        all_words.extend(words)
    word_list=nltk.FreqDist(all_words)
    word_features = wordlist.key()
    return word_features

def extractFeatures(tweet):
    tweet_words = set( tweet )
    features={}
    for word in word_features:
        features['contains %s' %word] = (word in tweet_words)
    return features

word_features = buildVocabulary(ppTrainingData)
trainingFeatures=nltk.classify.apply_features(extract_features,ppTrainingData)
     ##   apply_features(feature_func, toks, labeled=None)
     ##   [(feature_func(tok), label) for (tok, label) in toks]
     ##  Feature vector is created inthis process in such a way that the words 
     ## present in the tweet are marked 1 and those that are not present are marked 
     ## 0, with there key in the form of "contain%s".
     
## applying NaiveBayesClassifier to classify train the training data
     
NBayesClassifier = nltk.NaiveBayesClassifier.train(trainingFeatures)

#####################################################################
# SVM classifier ####################################################
#####################################################################

from nltk.corpus import sentiwordnet as swn
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer 
##The sklearn.feature_extraction module deals with feature extraction from raw 
##data. It currently includes methods to extract features from text and images.

# We have to change the form of the data slightly. SKLearn has a CountVectorizer object. 
# It will take in documents and directly return a term-document matrix with the frequencies of 
# a word in the document. It builds the vocabulary by itself. We will give the trainingData 
# and the labels separately to the SVM classifier and not as tuples. 

svmTrainingData=[' '.join(tweet[0]) for tweet in ppTrainingData]
# Creates sentences out of the lists of words 

vectorizer=CountVectorizer(min_df=1)
X=vectorizer.fit_transform(svmTrainingData).toarray()
# We now have a term document matrix 
vocabulary=vectorizer.get_feature_names()

swn_weights=[]

for word in vocabulary:
    try:
        # Look for the synsets of that word in sentiwordnet 
        synset=list(swn.senti_synsets(word))
        # use the first synset only to compute the score, as this represents the most common 
        # usage of that word 
        common_meaning =synset[0]
        # If the pos_Score is greater, use that as the weight, if neg_score is greater, use -neg_score
        # as the weight 
        if common_meaning.pos_score()>common_meaning.neg_score():
            weight=common_meaning.pos_score()
        elif common_meaning.pos_score()<common_meaning.neg_score():
            weight=-common_meaning.neg_score()
        else: 
            weight=0
    except: 
        weight=0
    swn_weights.append(weight)
    
swn_X=[]
for row in X:
    swn_X.append(np.multiply(row,np.array(swn_weights)))
swn_X=np.vstack(swn_X)

labels_to_array={"positive":1,"negative":2}
labels=[labels_to_array[tweet[1]] for tweet in ppTrainingData]
y=np.array(labels)

from sklearn.svm import SVC 
SVMClassifier=SVC()
SVMClassifier.fit(swn_X,y)

##############################################################################
## Running Classifier for 1st model ##########################################
##############################################################################
NBResultLabels=[NBayesClassifier.classify(extract_features(tweet[0])) for tweet in ppTestData]

##############################################################################
## Running Classifier for 2nd model ##########################################
##############################################################################

SVMResultLabels=[]
for tweet in ppTestData:
    tweet_sentence=' '.join(tweet[0])
    svmFeatures=np.multiply(vectorizer.transform([tweet_sentence]).toarray(),np.array(swn_weights))
    SVMResultLabels.append(SVMClassifier.predict(svmFeatures)[0])

##############################################################################
## Result from Models ########################################################
##############################################################################
    

if NBResultLabels.count('positive')>NBResultLabels.count('negative'):
    print "NB Result Positive Sentiment" + str(100*NBResultLabels.count('positive')/len(NBResultLabels))+"%"
else: 
    print "NB Result Negative Sentiment" + str(100*NBResultLabels.count('negative')/len(NBResultLabels))+"%"

if SVMResultLabels.count(1)>SVMResultLabels.count(2):
    print "SVM Result Positive Sentiment" + str(100*SVMResultLabels.count(1)/len(SVMResultLabels))+"%"
else: 
    print "SVM Result Negative Sentiment" + str(100*SVMResultLabels.count(2)/len(SVMResultLabels))+"%"
  


        
        
    
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            