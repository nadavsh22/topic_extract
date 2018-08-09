###################-IMPORTS-##########################
import nltk
import feedparser
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
import nltk.data
import sys

#######################################################


###################-CONSTANTS-#########################
DEFAULT_URL = "http://feeds.feedburner.com/TechCrunch/"
cut_from_end = -9
cut_from_start = 0
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
html_regex = '<.*?>'
qoute_pattern = r'"(.*?)"'
NOUNS = {'NN', 'NNP'}

#######################################################

###################-Define CFG-#########################

#######################################################

def main_func(feed_url):
    feed = feedparser.parse(feed_url)
    for entry in feed.entries:
        print(entry.title)
        print(entry.link)
        sentences = preprocess_content(entry)
        run_LDA(sentences)


def run_LDA(sentences):
    dictionary = corpora.Dictionary(sentences)
    term_matrix = [dictionary.doc2bow(sentence) for sentence in sentences]
    LDA = gensim.models.ldamodel.LdaModel
    ldamodel = LDA(term_matrix, num_topics=10, id2word=dictionary, passes=50)
    results = ldamodel.show_topics(num_topics=10, num_words=1)
    words = set()
    for topic in results:
        words.add(re.findall(qoute_pattern, topic[1])[0])
    # print(ldamodel.print_topics(num_topics=3, num_words=2))
    # print(ldamodel.print_topics(num_topics=2, num_words=3))
    words = nltk.pos_tag(words)
    words = [word[0] for word in words if word[1] in NOUNS]
    print(words)


def preprocess_content(entry):
    entry_content = re.sub(html_regex, '', entry.content[0].get('value'))
    tokenaizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = [single_doc_clean(sentence).split() for sentence in tokenaizer.tokenize(entry_content)]
    return sentences


def single_doc_clean(doc):
    no_stop = " ".join([word for word in doc.lower().split() if word not in stop])
    no_punc = ''.join(char for char in no_stop if char not in exclude and not char.isdigit())
    tokenz = " ".join(lemma.lemmatize(word) for word in no_punc.split() if len(word) >= 3)
    return tokenz


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main_func(DEFAULT_URL)
    else:
        main_func(sys.argv[1])
