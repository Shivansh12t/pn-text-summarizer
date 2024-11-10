import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import string
import re

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """Preprocess text by cleaning and tokenizing sentences and words, removing stopwords, and stemming."""
    # Replace bullet points and excessive newlines with a period for sentence tokenization
    text = re.sub(r'•|\n', '. ', text)
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    preprocessed_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [w.lower() for w in words if w.lower() not in stop_words and w not in string.punctuation]
        preprocessed_sentences.append(words)
    
    return sentences, preprocessed_sentences

def compute_tf(preprocessed_sentences):
    """Compute term frequency (TF) of words in preprocessed sentences."""
    tf_score = defaultdict(int)
    for sentence in preprocessed_sentences:
        for word in sentence:
            tf_score[word] += 1
    return tf_score

def score_sentences(sentences, preprocessed_sentences, tf_score):
    """Score sentences based on term frequencies."""
    sentence_scores = {}
    for i, words in enumerate(preprocessed_sentences):
        sentence_scores[i] = sum(tf_score[word] for word in words) / (len(words) + 1e-5)  # Prevent division by zero
    return sentence_scores

def summarize(text, top_n=3):
    """Summarize text by extracting top-n highest-scoring sentences."""
    sentences, preprocessed_sentences = preprocess_text(text)
    tf_score = compute_tf(preprocessed_sentences)
    sentence_scores = score_sentences(sentences, preprocessed_sentences, tf_score)
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:top_n]
    summary = [sentences[i] for i in sorted(top_sentences)]
    return ' '.join(summary)

if __name__ == "__main__":
    # Input text
    text = (
        """
A transition to a complex state is equivalent to a
simultaneous transition to the initial states of
each concurrent statechart
• An initial state must be specified in both nested
statecharts in order to avoid ambiguity about
which substate should first be entered in each
concurrent region
• A transition to the Active state means that the
Campaign object simultaneously enters the
Advert Preparation and Survey states
        """
    )
    
    # Summarize the text
    summary = summarize(text, top_n=2)
    print("Summary:")
    print(summary)
