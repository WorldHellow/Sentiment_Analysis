import nltk
import random
from nltk.corpus import movie_reviews

import pickle

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

#print(documents[1])

words = []

for w in movie_reviews.words():
    words.append(w.lower())

words = nltk.FreqDist(words)
#print (words)
#print(words["stupid"])

features = list(words.keys())[:3000]

def look_for_features(document):
    w = set(document)
    f = {}
    for x in features:
        f[x] = {x in w}
    return f

#find features of negative dataset
#print((look_for_features(movie_reviews.words('neg/cv000_29416.txt'))))

#feature set will be finding features and category
featuresets = [[look_for_features(rev), category] for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

##classifier_f = open("naivebayes.picke","rb")
##classifier = pickle.load(classifier_f)
##classifier_f.close()

print ("Accuracy: ", (nltk.classify.accuracy(classifier,testing_set))*100)

#most common words
classifier.show_most_informative_features(10)

save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()
