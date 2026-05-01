#!/usr/bin/env python
import re, random, math, collections, itertools

PRINT_ERRORS=0

#------------- Function Definitions ---------------------


def readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia):

    #reading pre-labeled input and splitting into lines
    posSentences = open('rt-polarity.pos', 'r', encoding="ISO-8859-1")
    posSentences = re.split(r'\n', posSentences.read())

    negSentences = open('rt-polarity.neg', 'r', encoding="ISO-8859-1")
    negSentences = re.split(r'\n', negSentences.read())

    posSentencesNokia = open('nokia-pos.txt', 'r')
    posSentencesNokia = re.split(r'\n', posSentencesNokia.read())

    negSentencesNokia = open('nokia-neg.txt', 'r', encoding="ISO-8859-1")
    negSentencesNokia = re.split(r'\n', negSentencesNokia.read())

    with open('positive-words.txt', 'r', encoding="ISO-8859-1") as posDictionary:
        posWordList = []
        for line in posDictionary:
            if not line.startswith(';'):
                posWordList.extend(re.findall(r"[a-z\-]+", line))
    posWordList.remove('a')

    with open('negative-words.txt', 'r', encoding="ISO-8859-1") as negDictionary:
        negWordList = []
        for line in negDictionary:
            if not line.startswith(';'):
                negWordList.extend(re.findall(r"[a-z\-]+", line))

    for i in posWordList:
        sentimentDictionary[i] = 1
    for i in negWordList:
        sentimentDictionary[i] = -1

    #create Training and Test Datsets:
    #We want to test on sentences we haven't trained on, to see how well the model generalses to previously unseen sentences

  #create 90-10 split of training and test data from movie reviews, with sentiment labels    
    for i in posSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="positive"
        else:
            sentencesTrain[i]="positive"

    for i in negSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="negative"
        else:
            sentencesTrain[i]="negative"

    #create Nokia Datset:
    for i in posSentencesNokia:
            sentencesNokia[i]="positive"
    for i in negSentencesNokia:
            sentencesNokia[i]="negative"

#----------------------------End of data initialisation ----------------#

#calculates p(W|Positive), p(W|Negative) and p(W) for all words in training data
def trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord):
    posFeatures = [] # [] initialises a list [array]
    negFeatures = [] 
    freqPositive = {} # {} initialises a dictionary [hash function]
    freqNegative = {}
    dictionary = {}
    posWordsTot = 0
    negWordsTot = 0
    allWordsTot = 0

    #iterate through each sentence/sentiment pair in the training data
    for sentence, sentiment in sentencesTrain.items():
        wordList = re.findall(r"[\w']+", sentence)
        
        for word in wordList: #calculate over unigrams
            allWordsTot += 1 # keeps count of total words in dataset
            if not (word in dictionary):
                dictionary[word] = 1
            if sentiment=="positive" :
                posWordsTot += 1 # keeps count of total words in positive class

                #keep count of each word in positive context
                if not (word in freqPositive):
                    freqPositive[word] = 1
                else:
                    freqPositive[word] += 1    
            else:
                negWordsTot+=1# keeps count of total words in negative class
                
                #keep count of each word in positive context
                if not (word in freqNegative):
                    freqNegative[word] = 1
                else:
                    freqNegative[word] += 1

    for word in dictionary:
        #do some smoothing so that minimum count of a word is 1
        if not (word in freqNegative):
            freqNegative[word] = 1
        if not (word in freqPositive):
            freqPositive[word] = 1

        # Calculate p(word|positive)
        pWordPos[word] = freqPositive[word] / float(posWordsTot)

        # Calculate p(word|negative) 
        pWordNeg[word] = freqNegative[word] / float(negWordsTot)

        # Calculate p(word)
        pWord[word] = (freqPositive[word] + freqNegative[word]) / float(allWordsTot) 

#---------------------------End Training ----------------------------------
#implement naive bayes algorithm
#INPUTS:
#  sentencesTest is a dictonary with sentences associated with sentiment 
#  dataName is a string (used only for printing output)
#  pWordPos is dictionary storing p(word|positive) for each word
#     i.e., pWordPos["apple"] will return a real value for p("apple"|positive)
#  pWordNeg is dictionary storing p(word|negative) for each word
#  pWord is dictionary storing p(word)
#  pPos is a real number containing the fraction of positive reviews in the dataset
def testBayes(sentencesTest, dataName, pWordPos, pWordNeg, pWord,pPos):
    pNeg=1-pPos

    #These variables will store results
    total=0
    correct=0
    totalpos=0
    totalpospred=0
    totalneg=0
    totalnegpred=0
    correctpos=0
    correctneg=0

    #for each sentence, sentiment pair in the dataset
    for sentence, sentiment in sentencesTest.items():
        wordList = re.findall(r"[\w']+", sentence)#collect all words

        pPosW=pPos
        pNegW=pNeg

        for word in wordList: #calculate over unigrams
            if word in pWord:
                if pWord[word]>0.00000001:
                    pPosW *=pWordPos[word]
                    pNegW *=pWordNeg[word]

        prob=0;            
        if pPosW+pNegW >0:
            prob=pPosW/float(pPosW+pNegW)


        total+=1
        if sentiment=="positive":
            totalpos+=1
            if prob>0.5:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("ERROR (pos classed as neg %0.2f):" %prob + sentence)
        else:
            totalneg+=1
            if prob<=0.5:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    print ("ERROR (neg classed as pos %0.2f):" %prob + sentence)
 
 
# TODO for Step 2: Add some code here to calculate and print: (1) accuracy; (2) precision and recall for the positive class; 
# (3) precision and recall for the negative class; (4) F1 score;
    accuracy = 0
    prepos = 0
    recpos = 0
    preneg = 0
    recneg = 0
    f1pos = 0
    f1neg = 0

    accuracy = (correct / total) 

    #Precision = positive
    #(TP) / (TP+FP)
    prepos = ((correctpos) / (totalpospred)) 
    recpos = ((correctpos) / (totalpos))

    #Recall = Negative
    preneg = ((correctneg) / (totalnegpred)) 
    recneg = ((correctneg) / (totalneg)) 

    #F1 Score = F1=2×( Precision×Recall / Precision+Recall )
    f1pos = (2 * ((prepos * recpos) / (prepos + recpos))) 
    f1neg = (2 * ((preneg * recneg) / (preneg + recneg))) 


    print(f'Accuracy: {accuracy*100:.2f} %')
    print(f'Precision (positive): {prepos*100:.2f} %')
    print(f'Recall (Positive): {recpos*100:.2f} %')
    print(f'Precision (Negative): {preneg*100:.2f} %')
    print(f'Recall (Negative): {recneg*100:.2f} %')

    print(f'F1 Score (Positive) {f1pos:.2f}')
    print(f'F1 Score (Negative) {f1neg:.2f}')

'''
We can find that the accuracy on the Sentencestrain (TrainingData) is around 88% whereas the accuracy on SentencesTest (TestData) is around 77%.
The model performs well on the training data with high accuracy, precision, recall and F1 measure.
On Test data, while performance is slightly lower, the model still maintains a good balance between precision and recall as indicated by the F1 measure.
The drop in performance from training to test indicates the possibility of overfitting or differences in distribution of sentiments between the two datasets.
Further analysis and potentially adjusting the model parameters could be explored to improve generalization to unseen data. 
'''


# This is a simple classifier that uses a sentiment dictionary to classify 
# a sentence. For each word in the sentence, if the word is in the positive 
# dictionary, it adds 1, if it is in the negative dictionary, it subtracts 1. 
# If the final score is above a threshold, it classifies as "Positive", 
# otherwise as "Negative"

def testDictionary(sentencesTest, dataName, sentimentDictionary, threshold):
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0

    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        for word in Words:
            if word in sentimentDictionary:
               score+=sentimentDictionary[word]
 
        total+=1
        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1

# TODO for Step 5: Add some code here to calculate and print: (1) accuracy; (2) precision and recall for the positive class; 
# (3) precision and recall for the negative class; (4) F1 score;
    accuracy = 0
    prepos = 0
    recpos = 0
    preneg = 0
    recneg = 0
    f1pos = 0
    f1neg = 0

    accuracy = (correct / total) 

    #Precision = positive
    #(TP) / (TP+FP)
    prepos = ((correctpos) / (totalpospred)) 
    recpos = ((correctpos) / (totalpos))

    #Recall = Negative
    preneg = ((correctneg) / (totalnegpred)) 
    recneg = ((correctneg) / (totalneg)) 

    #F1 Score = F1=2×( Precision×Recall / Precision+Recall )
    f1pos = (2 * ((prepos * recpos) / (prepos + recpos))) 
    f1neg = (2 * ((preneg * recneg) / (preneg + recneg))) 


    print(f'Accuracy: {accuracy*100:.2f} %')
    print(f'Precision (positive): {prepos*100:.2f} %')
    print(f'Recall (Positive): {recpos*100:.2f} %')
    print(f'Precision (Negative): {preneg*100:.2f} %')
    print(f'Recall (Negative): {recneg*100:.2f} %')

    print(f'F1 Score (Positive) {f1pos:.2f}')
    print(f'F1 Score (Negative) {f1neg:.2f}')

#  sentencesTest is a dictonary with sentences associated with sentiment 
#  dataName is a string (used only for printing output)
#  pWordPos is dictionary storing p(word|positive) for each word
#     i.e., pWordPos["apple"] will return a real value for p("apple"|positive)
#  pWordNeg is dictionary storing p(word|negative) for each word
#  pWord is dictionary storing p(word)
#  pPos is a real number containing the fraction of positive reviews in the dataset

def improve_rule_based(sentencesTest, dataName, sentimentDictionary, threshold, imp_words):
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0


    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score = 0
        sentiment_dict = dict(sentimentDictionary)
        negation_flag = False

        negative_words = ['no', 'not', 'never', "but", "however", "although"]
        diminisher_words = ["little", "few", "somewhat", "barely", "hardly","slightly"]
        emphasis_words = ["very", "extremely", "incredibly", "highly", "quite","immensely","intensely","hugely"]
        intensifier_words = ['absolutely', 'utterly', 'completely', 'totally', 'nearly', 'virtually', 'essentially','mainly', 'almost']

        for i,word in enumerate(Words):         

            if word.isnumeric():
                continue
            else:
                if word in negative_words:
                    for j in range(-3,3) :
                        if (i+j) < len(Words) and i+j >= 0 and Words[i+j] in sentimentDictionary:
                            if sentimentDictionary[Words[i+j]] == 1:
                                #word_score = sentimentDictionary[word]
                                #score += -word_score if negation_flag else word_score
                                score -= 2
                if word.isupper():
                    score *= 1.2

                if word.lower() in diminisher_words:
                    score *= 0.5

                if word.lower() in emphasis_words:
                    score *= 1.2

                if word.lower() in intensifier_words:
                    score *= 1.2 

                if word in imp_words:
                    score *= 1.5

                if word in sentimentDictionary:
                    score+=sentimentDictionary[word]

        total+=1
        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
    
# TODO for Step 5: Add some code here to calculate and print: (1) accuracy; (2) precision and recall for the positive class; 
# (3) precision and recall for the negative class; (4) F1 score;
    accuracy = 0
    prepos = 0
    recpos = 0
    preneg = 0
    recneg = 0
    f1pos = 0
    f1neg = 0

    accuracy = (correct / total) 

    #Precision = positive
    #(TP) / (TP+FP)
    prepos = ((correctpos) / (totalpospred)) 
    recpos = ((correctpos) / (totalpos))

    #Recall = Negative
    preneg = ((correctneg) / (totalnegpred)) 
    recneg = ((correctneg) / (totalneg)) 

    #F1 Score = F1=2×( Precision×Recall / Precision+Recall )
    f1pos = (2 * ((prepos * recpos) / (prepos + recpos))) 
    f1neg = (2 * ((preneg * recneg) / (preneg + recneg))) 


    print(f'Accuracy: {accuracy*100:.2f} %')
    print(f'Precision (positive): {prepos*100:.2f} %')
    print(f'Recall (Positive): {recpos*100:.2f} %')
    print(f'Precision (Negative): {preneg*100:.2f} %')
    print(f'Recall (Negative): {recneg*100:.2f} %')

    print(f'F1 Score (Positive) {f1pos:.2f}')
    print(f'F1 Score (Negative) {f1neg:.2f}')


#Print out n most useful predictors
def mostUseful(pWordPos, pWordNeg, pWord, n):
    predictPower={}
    for word in pWord:
        if pWordNeg[word]<0.0000001:
            predictPower[word]=1000000000
        else:
            predictPower[word]=pWordPos[word] / (pWordPos[word] + pWordNeg[word])
            
    sortedPower = sorted(predictPower, key=predictPower.get)
    head, tail = sortedPower[:n], sortedPower[len(predictPower)-n:]
    #print ("NEGATIVE:")
    #print (head)
    #print ("\nPOSITIVE:")
    #print (tail)

    imp_words = head + tail
    count = 0
    uncommon = 0
    for word in imp_words:
        if word in sentimentDictionary:
            count += 1
        else:
            uncommon += 1
            
    #print(f'Similarity Percentage with Imp_words {(count/len(imp_words))*100:.2f}')
    print('The number of words common in Sentiment Dictionary is : ', count)
    #print('Uncommon word count: ', uncommon)
    #print('Length of Sentiment Dictionary: ', len(sentimentDictionary))

    return imp_words

'''
The function calculates the PredictPower for each word and sorts them in ascending order according to the power, placing the .
It is important to note that the model's understanding of sentiment is based on the training data it was exposed to.
The selected words are those that the model has learned are useful indicators of sentiment, but this does not necessarily mean that they universally represent "good" or "bad" sentiments.
the words selected by the model are considered as good indicators of sentiment based on their calculated predictPower values. The model believes that these words contribute significantly
to predicting whether a given text expresses positive or negative sentiment. However, the effectiveness of these words in capturing sentiment nuances depends on the training data and
the specific context of the sentiment analysis task.

Limited Coverage of Important Words:

The sentiment dictionary has a limited coverage of the important words that were identified as most useful in determining sentiment. This means that a significant portion of words that could contribute to sentiment analysis may not be captured by the dictionary.
Reduced Precision and Recall:

The precision and recall of the sentiment dictionary for sentiment analysis may be compromised. Precision measures the accuracy of positive/negative predictions among the words in the dictionary, while recall measures the ability to capture all positive/negative words. With only 45% coverage, both precision and recall are likely to be suboptimal.
Incomplete Sentiment Understanding:

The sentiment dictionary may not have a comprehensive understanding of the diverse language used in sentences. Sentiment analysis benefits from a wide-ranging lexicon that includes words expressing nuanced sentiment, and the limited coverage suggests that the dictionary might miss crucial sentiment indicators.
Potential for Misclassifications:

The sentiment dictionary may be prone to misclassifications due to the absence of important words. Sentences containing crucial sentiment-bearing words that are not present in the dictionary may be incorrectly classified, leading to less accurate sentiment predictions.
'''
#---------- Main Script --------------------------


sentimentDictionary={} # {} initialises a dictionary [hash function]
sentencesTrain={}
sentencesTest={}
sentencesNokia={}

#initialise datasets and dictionaries
readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia)

pWordPos={} # p(W|Positive)
pWordNeg={} # p(W|Negative)
pWord={}    # p(W) 

#build conditional probabilities using training data
trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord)
#trainBayes(sentencesNokia, pWordPos, pWordNeg, pWord)

#run naive bayes classifier on datasets
print ("Naive Bayes")

#testBayes(sentencesTrain,  "Films (Train Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
#testBayes(sentencesTest,  "Films  (Test Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
#testBayes(sentencesNokia, "Nokia   (All Data,  Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.7)


#run sentiment dictionary based classifier on datasets
testDictionary(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionary, 1)
#testDictionary(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, 1.4)
#testDictionary(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionary, 0.6)

#print('The most useful words:')
imp_words = mostUseful(pWordPos, pWordNeg, pWord, 100)

improve_rule_based(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionary, 1,imp_words)
#improve_rule_based(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, 1.4,imp_words)
#improve_rule_based(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionary, 0.6,imp_words)

#C:\Sheffield\TextProcessing\Code_data_V1-2\Code_data_V1-2\Sentiment.py
