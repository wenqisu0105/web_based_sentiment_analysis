
import regex as re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet



# Define a function to convert POS tags to WordNet POS tags
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
    
# Define a function to tokenize, POS tag, and lemmatize a list of tokens
def tokenize_pos_lemmatize(tokens):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    pos_tags = nltk.pos_tag(tokens)
    lemmas = []
    lemma_pos = []
    for token, pos_tag in pos_tags:
        wordnet_pos = get_wordnet_pos(pos_tag)
        lemma = lemmatizer.lemmatize(token, pos=wordnet_pos)
        lemmas.append(lemma)
        lemma_pos.append((lemma, wordnet_pos))
    return lemmas, lemma_pos


def process_input(text):
    text = text.lower()
    text = text.lower()
    text = re.sub(r"https?\S+", '', text)
    tknzr = TweetTokenizer()
    tokens = tknzr.tokenize(text)
    lemma_tokens, tokens_pos = tokenize_pos_lemmatize(tokens)
    return lemma_tokens
