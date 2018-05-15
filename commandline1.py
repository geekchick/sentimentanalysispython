import nltk

# Implementation

# List of training tweets
my_tweets = [('I love this car', 'positive'),
              ('This view is amazing', 'positive'),
              ('I feel great this morning', 'positive'),
              ('I am so excited about the concert', 'positive'),
              ('He is my best friend', 'positive'),
              ('I do not like this car', 'negative'),
              ('This view is horrible', 'negative'),
              ('I feel tired this morning', 'negative'),
              ('I am not looking forward to the concert', 'negative'),
              ('He is my enemy', 'negative')]

# List of test tweets
test_set =[("I feel happy this morning", 'positive'),
              ("Larry is my friend", 'positive'),
           ("I do not like that man", 'negative'),
           ("My house is not great", 'negative'),
           ("Your song is annoying", 'negative')]



# make the tweets all lowercase
def lowercaseTweets(tweets): # have to split first or else loops through the whole sentence infinity because appending lines
    # tweet = [] --> what happens? tweets get reset to none and prints out nothing
    # create new list
    tweets_lower = []
    for (words, sentiment) in tweets:
        lowercase_words = words.lower()
        #tweets.append((lowercase_words, sentiment)) ---> results in infinite loop
        tweets_lower.append((lowercase_words, sentiment))
        #print(tweets_lower)

    return tweets_lower

# Split the list by spaces
tweets = []
def splitList(tweets):
    # Create new list
    tweets_split = []
    # For every tuple in the list, split the values
    for (words, sentiment) in tweets:
        # create variable and split words
        split_word = words.split()
        #print("This is split word: ", split_word)
        my_tweets = []
        for (word) in split_word:
            if len(word) >= 3:
                my_tweets.append((word))
            #print(my_tweets)
        # append to new list if word is greater than 3 characters
        tweets_split.append((my_tweets, sentiment))

        #print(split_word)

    return tweets_split


# can't do this:
# word_features = lowercaseTweets(splitList(training_set))) --> error can't user lower on list
new_list = splitList(lowercaseTweets((my_tweets)))
print(new_list)

# CLASSIFIER

# List of words needs to be extracted from tweets
# This new list contains every distinct word ordered by frequency of appearance

# Takes all the tweets and puts them in 1 list using list.extend()
def get_words_in_tweets(tweets):
    # create an empty list
    all_words = []
    # Loop through every tweet and sentiment
    for (words, sentiment) in tweets:
        #print("This is word: ", words)
        # extend each tweet to the list all_words
        all_words.extend(words)
        #print("This is list all_words: ", all_words)
    # returns all of the tweets in the list
    return all_words

def get_word_features(wordlist):
    # Use nltk to get the FreqDist
    wordlist = nltk.FreqDist(wordlist)
    #print(wordlist)
    # Get the most commonly used words with a count for each
    common = wordlist.most_common(20)
    #print(type(common))
    #print(common)
    # Add the most common words to a list named common_list
    # Create an empty list
    common_list = []
    # Set a counter to 0
    i = 0
    # Loop through each item in common as long as the position is less than the length of the list
    for item in range(len(common)):
        # Create a variable and put the first item in the tuple in it
        list_item = common[i][0]
        # Append that item to the list
        common_list.append(list_item)
        # Go to the next position
        i = i + 1
    #word_features = wordlist.keys()
    #print(common_list)
    return common_list

#print(get_words_in_tweets(new_list))

new_word_features = get_word_features(get_words_in_tweets(new_list))
print(new_word_features)
#print(get_word_features(new_list))


# To create a classifier we need to decide what features are relevant
# Need a Feature Extractor

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in new_word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

#print(extract_features(['love', 'this', 'car']))

# Apply the features to our classifier using apply_features
# pass the features extractor along with the tweets (new_list)
training_set = nltk.classify.apply_features(extract_features, new_list)
print(training_set)

# Train our classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)

# Classify a tweet
tweet = 'I do not like that man'
classify_tweet = classifier.classify(extract_features(tweet.split()))
print(classify_tweet)
