import re
from nltk.corpus import cmudict
from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

# uncomment the following if you do not have these lexicons

# nltk.download('cmudict')
# nltk.download('wordnet')

p_dict = cmudict.dict()

class TextClassifier(object):

    def __init__(self, text: str):
        """
        Sets some class wide variables.
        """

        self.__text = text
        self.__lemmatizer = WordNetLemmatizer()
        self.__vowels = {"a", "e", "i", "o", "u", "y"}

    def __lemmatize(self, word):
        """
        Uses WordNetLemmatizer to lemmatize a word

        :param
            word: a word to lemmatize
        :return:
            A lemma
        """

        lemma = self.__lemmatizer.lemmatize(word, 'v')
        if lemma == word:
            lemma = self.__lemmatizer.lemmatize(word, 'n')
        return lemma

    def __preprocess_text(self, is_sent=False, tagged=False, lemmatized=False):
        """
        Preprocesses the text, by applying POS tagging and lemmatization depending on the specified parameters set.
        Yields a sentence or a word

        :params
            is_sent: a check whether it is a sentence or a word
            tagged: a switch whether the text should be tagged or not
            lemmatized: a switch whether the text should be lemmatized or not
        """

        for sent in sent_tokenize(self.__text):
            tokenized_sent = [word.strip() for word in word_tokenize(sent) if word.strip()]

            if tagged:
                tokenized_sent = pos_tag(tokenized_sent)
                if lemmatized:
                    tokenized_sent = list(map(lambda x: (self.__lemmatize(x[0]), x[1]), tokenized_sent))
            elif lemmatized:
                tokenized_sent = list(map(self.__lemmatize, tokenized_sent))

            if is_sent:
                yield tokenized_sent
            else:
                for word in tokenized_sent:
                    yield word

    def sents(self, lemmatized=False):
        """
        Returns an array of sentences

        :param
            lemmatized: a switch whether the sentences should be lemmatized or not
        :return:
            A list of sentences
        """

        return self.__preprocess_text(is_sent=True, tagged=False, lemmatized=lemmatized)

    def tagged_sents(self, lemmatized=False):
        """
        Returns an array of POS tagged sentences

        :param
            lemmatized: a switch whether the sentences should be lemmatized or not
        :return:
            A list of POS tagged sentences
        """

        return self.__preprocess_text(is_sent=True, tagged=True, lemmatized=lemmatized)

    def words(self, lemmatized=False):
        """
        Returns an array of words

        :param
            lemmatized: a switch whether the sentences should be lemmatized or not
        :return:
            A list of words
        """

        return self.__preprocess_text(is_sent=False, tagged=False, lemmatized=lemmatized)

    def tagged_words(self, lemmatized=False):
        """
        Returns an array of POS tagged words

        :param
            lemmatized: a switch whether the sentences should be lemmatized or not
        :return:
            A list of POS tagged words
        """

        return self.__preprocess_text(is_sent=False, tagged=True, lemmatized=lemmatized)

    def has_peculiar_expression(self, expression):
        """
        Checks whether the text contains some expression of interest such as a regional greeting

        :param
            expression: the peculiar expression
        :return:
            A boolean
        """

        return bool(re.search(r"" + expression + "", self.__text, re.IGNORECASE))

    def count_syllables(self, word: str):
        """
        use CMU dict (p_dict) to count the number of
        syllables in word, default to number of vowels
        """

        word = word.lower()
        if word in p_dict:
            iterator = filter(lambda phone: phone[-1].isdigit(), p_dict[word][0])
        else:
            iterator = filter(lambda letter: letter in self.__vowels, list(word))

        return len(list(iterator))

    def calculate_sentence_reading_ease(self):
        """
        Calculate the Flesh reading ease for a single sentence

        :return:
            A readability score
        """

        reading_ease = 0.0
        for sent in self.sents():
            words = [word for word in sent if word.isalpha()]
            if words:
                syllables = [self.count_syllables(word) for word in words]
                reading_ease += (206.835 - (1.015 * len(words)) - 84.6 * (sum(syllables) / len(words)))

        return reading_ease

    def calculate_lexical_density_by_tags(self, tags_to_search: set):
        """
        Calculates the lexical density of words tagged by a list of specified open classes

        :param
            tags_to_search: a list of open class POS tags
        :return:
            A lexical density value
        """
        total_tags = sum(1 for word, tag in self.tagged_words() if tag in tags_to_search)
        total_words = len(list(self.tagged_words()))

        return round(total_tags / total_words * 100, 3)

    def calculate_words_frequency(self, words_to_search: set):
        """
        Calculates the frequency of given words in text

        :param
            words_to_search: a list of words for calculating frequency
        :return:
            Word frequency
        """

        total_peculiar_words = sum(1 for word in self.words() if word in words_to_search)
        total_words = len(list(self.words()))

        return round(total_peculiar_words / total_words * 1000, 3)

    def calculate_type_token_ratio(self):
        """
        Calculates the type token ratio (TTR) of a given text

        :return:
            the TTR value
        """

        word_list = [word.lower() for word in self.words(lemmatized=True)]
        return len(set(word_list)) / len(word_list)
