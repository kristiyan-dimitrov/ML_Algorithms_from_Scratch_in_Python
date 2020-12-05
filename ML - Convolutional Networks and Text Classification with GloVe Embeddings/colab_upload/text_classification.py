import numpy as np
# import sklearn

from collections import Counter

from src.preprocess import *
from src.run_torch_model import *
from src.text_model import *
from src.my_dataset import *



class TextClassificationModel:
    def __init__(self):
        pass

    def train(self, texts, labels):
        """
        Trains the model.  The texts are raw strings, so you will need to find
        a way to represent them as feature vectors to apply most ML methods.

        You can implement this using any ML method you like.  You are also
        allowed to use third-party libraries such as sklearn, scipy, nltk, etc,
        with a couple exceptions:

        - The classes in sklearn.feature_extraction.text are *not* allowed, nor
          is any other library that provides similar functionality (creating
          feature vectors from text).  Part of the purpose of this project is
          to do the input featurization yourself.  You are welcome to look
          through sklearn's documentation for text featurization methods to get
          ideas; just don't import them.  Also note that using a library like
          nltk to split text into a list of words is fine.

        - An exception to the above exception is that you *are* allowed to use
          pretrained deep learning models that require specific featurization.
          For example, you might be interested in exploring pretrained
          embedding methods like "word2vec", or perhaps pretrained models like
          BERT.  To use them you have to use the same input features that the
          creators did when pre-training them, which usually means using the
          featurization code provided by the creators.  The rationale for
          allowing this is that we want you to have the opportunity to explore
          cutting-edge ML methods if you want to, and doing so should already
          be enough work that you don't need to also bother with doing
          featurization by hand.

        - When in doubt, ask an instructor or TA if a particular library
          function is allowed or not.

        Hints:
        - Don't reinvent the wheel; a little reading on what techniques are
          commonly used for featurizing text can go a long way.  For example,
          one such method (which has many variations) is TF-IDF:
          https://en.wikipedia.org/wiki/Tf-idf
          https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System

        - There are multiple ways to complete the assignment.  With the right
          featurization strategy, you can pass the basic tests with one of the
          ML algorithms you implemented for the previous homeworks.  To pass
          the extra credit tests, you may need to use torch or sklearn unless
          your featurization is exceptionally good or you make some special
          modifications to your previous homework code.

        Arguments:
            texts - A list of strings representing the inputs to the model
            labels - A list of integers representing the class label for each string
        Returns:
            Nothing (just updates the parameters of the model)
        """

        preprocessed_texts = preprocess_texts(texts)

        self.vocabulary = self._non_rare_vocabulary(preprocessed_texts)

        # BAG OF WORDS
        # features = self._bag_of_words(preprocessed_texts, self.vocabulary)

        # TFIDF
        features = self._tf_idf(preprocessed_texts, self.vocabulary)

        print("FEATURES.shape ", features.shape)

        train_dataset = MyDataset(features, labels)

        print("len(train_dataset) ", len(train_dataset))

        self.model = Text_Classifier(input_size = len(self.vocabulary)
                                    , output_size = len(np.unique(labels)))


        self.model, loss, accuracy = run_model(self.model
                                               , running_mode='train'
                                               , train_set = train_dataset
                                               , batch_size = 10
                                               , learning_rate = 1e-4
                                               , n_epochs = 3
                                               , shuffle = True)


    def predict(self, texts):
        """Predicts labels for the given texts.
        Arguments:
            texts - A list of strings
        Returns:
            A list of integers representing the corresponding class labels for the inputs
        """

        preprocessed_texts = preprocess_texts(texts)

        features = self._tf_idf(preprocessed_texts, self.vocabulary)

        test_dataset = MyDataset(features, np.zeros((len(features),))) # dummy labels

        test_dataloader = torch.utils.data.DataLoader(test_dataset
                                                        , batch_size=len(features) # Will just 1 batch work?
                                                        , shuffle=False
                                                        , num_workers=2) 

        with torch.no_grad():

            for batch, labels in test_dataloader:

                output = self.model(batch.float())

                predicted = torch.max(output.data, 1)[1]


        return predicted


    ######################
    ### HELPER METHODS ###
    ######################

    def _non_rare_vocabulary(self, texts, min_freq=20): # <------------ IMPORTANT PARAMETER

        counter = Counter(texts[0])

        for text in texts[1:]:
            counter += Counter(text)
        
        print("Size of vocabulary BEFORE removing rare tokens: ", len(counter))
        
        sorted_items = sorted(counter.items(), key = lambda x: x[1])
        
        index = 0
        
        for sorted_item in sorted_items:
            
            if sorted_item[1] >= min_freq:
                break
                
            index+=1
            
        print("Size of vocabulary AFTER  removing rare tokens: ", len(sorted_items[index:]))
            
        return [x[0] for x in sorted_items[index:]]


    def _bag_of_words(self, texts, vocabulary):
    
        bow = np.zeros((len(texts), len(vocabulary)))
        
        for ii in range(len(texts)):
            
            for word in texts[ii]:
                
                if word in vocabulary:
                    
                    bow[ii, vocabulary.index(word)] += 1
                    
        return bow


    def _term_document_frequency(self, term, texts):
    
        counter = 0
        
        for text in texts:
            if term in text:
                counter += 1
                
        return counter


    def _idf(self, texts, vocabulary):
        
        doc_freq = np.zeros((len(vocabulary),))
        N = len(texts)
                            
        for ii in range(len(vocabulary)):
            doc_freq[ii] = np.log(N / self._term_document_frequency(vocabulary[ii], texts))
            
        return doc_freq


    def _tf_idf(self, texts, vocabulary):
        
        # Instantiate results
        tfidf = np.zeros((len(texts), len(vocabulary)))
        
        # One-off calculation of Inverse Document Frequency
        doc_freq = self._idf(texts, vocabulary)
        
        for ii in range(len(texts)):
            # Find Term Frequency for each word in a text
            for word in texts[ii]:
                
                if word in vocabulary:
                    
                    tfidf[ii, vocabulary.index(word)] += 1
            
            # Multiply the Term Frequency by Inverse Document Frequency
            tfidf[ii, ] *= tfidf[ii, ] * doc_freq
                    
        return tfidf

