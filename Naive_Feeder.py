# for the top stories from techCrunch, prints their Link, title, and keywords
# currently prints keywords both from raw text and after wikipedia comparison
# note that tagging is done before input preprocessing, that is due to the fact
#  that capitalization is indicative of proper nouns, hence it yields more
# relevant results.

###################-IMPORTS-########################
import nltk
import feedparser
import re
import wikipedia
import sys

####################################################


###################-REGEX-##########################

word_pattern = r'\w{2,}'
word_pattern_regex = re.compile(word_pattern)
alpha_numeric = '[^A-Za-z0-9]+'
alpha_numeric_regex = re.compile(alpha_numeric)

#####################################################


###################-CONSTANTS-#######################

cut_from_end = -6
cut_from_start = 0
DIFF_THRESH = 0.5
DEFAULT_URL = "http://feeds.feedburner.com/TechCrunch/"
NAME = 0
TYPE = 1

######################################################


##################-Define Grammer-####################

GRAMMER = {"NNP+NNP": "NNP", "NN+NN": "NNS", "NNS+NN": "NNS", "JJ+JJ": "JJ", "JJ+NN": "NNS"}

######################################################

"""for all entries in given feed print link, title and keywords"""


def main_func(feed_url):
    feed = feedparser.parse(feed_url)
    for entry in feed.entries:
        print(entry.link)
        print(entry.title)
        print(', '.join(main_word_extract(entry.title, entry.summary)[0]))  # raw results
        print(', '.join(main_word_extract(entry.title, entry.summary)[1]))  # wikipedia comparison results


"""extract the main words of a sentence using the nltk tagger"""


def extractor(sentence):
    sentence = [word for word in sentence if not alpha_numeric_regex.match(word)]
    tags = nltk.pos_tag(sentence)
    merge = True
    while merge:
        merge = False
        for tag in range(len(tags) - 1):
            tag1 = tags[tag]
            tag2 = tags[tag + 1]
            key = "%s+%s" % (tag1[TYPE], tag2[TYPE])
            value = GRAMMER.get(key, '')
            if value:
                merge = True
                tags.pop(tag)
                tags.pop(tag)
                match = "%s %s" % (tag1[NAME], tag2[NAME])
                pos = value
                tags.insert(tag, (match, pos))
                break

    matches = []
    for tag in tags:
        if tag[TYPE] == "NNP" or tag[TYPE] == "NNI":
            matches.append(tag[NAME])
    return matches


"""receives title and summary of entry and return the main words"""


def main_word_extract(title, summary):
    summary = nltk.word_tokenize(summary)[cut_from_start:cut_from_end]
    title = nltk.word_tokenize(title)
    main_words_in_title = set(extractor(title))
    main_words_in_summary = set(extractor(summary))
    main_words_in_entry = main_words_in_title.union(main_words_in_summary)
    main_words_in_entry = [word.lower() for word in main_words_in_entry if
                           word_pattern_regex.match(word)]
    main_words_in_entry_filter = set()

    for word in main_words_in_entry:
        word = wiki_compare(word)
        main_words_in_entry_filter.add(word)

    return main_words_in_entry, main_words_in_entry_filter


"""receive a word/term and try to find a suggestion or search result on wikipedia"""


def wiki_compare(word):
    query_result = wikipedia.search(word, suggestion=True)
    if query_result[1] is None:
        search_result = query_result[0]
        if len(search_result) is not 0:
            search_result = search_result[0]
            diff = nltk.edit_distance(word, search_result) / max(len(search_result), len(word))
            if diff <= DIFF_THRESH:
                word = search_result
    else:
        word = query_result[1]
    return word


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main_func(DEFAULT_URL)
    else:
        main_func(sys.argv[1])
