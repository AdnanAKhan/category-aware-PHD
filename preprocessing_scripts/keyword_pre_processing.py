import os
import json
import re
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer


class ProcessKeywords:

    def __init__(self):
        """
        initiallization of data directories
        """
        self.DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')
        self.KEYWORDS_DIR = os.path.join(self.DATA_DIR, 'metadata', 'keywords')
        self.PROCESSED_KEYWORDS_DIR = os.path.join(self.DATA_DIR, 'metadata', 'processed_keywords')

    def process_single_file(self, file_name, function_list):
        keywords = json.load(open(os.path.join(self.KEYWORDS_DIR, file_name), 'r'))

        for func in function_list:
            keywords = func(keywords)

        processed_keywords = keywords

        return processed_keywords

    # list of pre-processing function
    @staticmethod
    def remove_stopwords(words):
        """Remove stop words from list of tokenized words"""
        new_words = []
        for word in words:
            if word not in stopwords.words('english'):
                new_words.append(word)
        return new_words

    @staticmethod
    def remove_non_ascii(words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)

        return new_words

    @staticmethod
    def to_lowercase(words):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)

        return new_words

    @staticmethod
    def remove_punctuation(words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)

        return new_words

    @staticmethod
    def remove_empty_strings(words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = word.strip()
            if len(new_word) > 0:
                new_words.append(new_word)
        return new_words

    def process(self):
        """
        process all files
        :return:
        """
        for file_name in os.listdir(self.KEYWORDS_DIR):
            pre_processing_functions = [ProcessKeywords.remove_non_ascii,
                                        ProcessKeywords.to_lowercase,
                                        ProcessKeywords.remove_punctuation,
                                        ProcessKeywords.remove_stopwords,
                                        ProcessKeywords.remove_empty_strings
                                        ]
            processed_keywords = self.process_single_file(file_name, pre_processing_functions)

            if not os.path.exists(self.PROCESSED_KEYWORDS_DIR):
                os.makedirs(self.PROCESSED_KEYWORDS_DIR)

            if len(processed_keywords) > 0:
                with open(os.path.join(self.PROCESSED_KEYWORDS_DIR, file_name), 'w') as f:
                    json.dump(processed_keywords, f)


if __name__ == '__main__':
    obj = ProcessKeywords()
    obj.process()
