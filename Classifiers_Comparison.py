import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

#print(documents[1])

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def look_for_features(document):
    words = set(document)
    features = {}
    for x in word_features:
        features[x] = x in words
    return features

#find features of negative dataset
#print((look_for_features(movie_reviews.words('neg/cv000_29416.txt'))))

#feature set will be finding features and category
featuresets = [(look_for_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1400]
testing_set = featuresets[1400:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

##classifier_f = open("naivebayes.picke","rb")
##classifier = pickle.load(classifier_f)
##classifier_f.close()

print ("Accuracy: ", (nltk.classify.accuracy(classifier,testing_set))*100)

#most common words
#classifier.show_most_informative_features(10)

#save_classifier = open("naivebayes.pickle","wb")
#pickle.dump(classifier, save_classifier)
#save_classifier.close()

#Multinomial
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print ("MNB_classifier Accuracy: ", (nltk.classify.accuracy(MNB_classifier,testing_set))*100)

#Bernoulli
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print ("BernoulliNB Accuracy: ", (nltk.classify.accuracy(BernoulliNB_classifier,testing_set))*100)

#Gaussian
#GaussianNB_classifier = SklearnClassifier(GaussianNB())
#GaussianNB_classifier.train(training_set)
#print ("GaussianNB Accuracy: ", (nltk.classify.accuracy(GaussianNB_classifier,testing_set))*100)

#Logistic Regression, SGD
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print ("LogisticRegression Accuracy: ", (nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print ("SGDClassifier Accuracy: ", (nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100)

#SVC, LinearSVC, NuSVC
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print ("SVC Accuracy: ", (nltk.classify.accuracy(SVC_classifier,testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print ("LinearSVC Accuracy: ", (nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print ("NuSVC Accuracy: ", (nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)

