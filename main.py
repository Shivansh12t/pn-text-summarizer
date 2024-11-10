import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import string

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """Preprocess text by tokenizing sentences and words, removing stopwords, and stemming."""
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
    # Example text
    text = (
        "Deep packet inspection (DPI) is a type of data processing that inspects in detail the data being sent "
        "over a computer network, and may take actions such as alerting, blocking, re-routing, or logging it accordingly. "
        "Deep packet inspection is often used for baselining application behavior, analyzing network usage, troubleshooting "
        "network performance, ensuring that data is in the correct format, checking for malicious code, eavesdropping, "
        "and internet censorship, among other purposes. There are multiple headers for IP packets; network equipment "
        "only needs to use the first of these (the IP header) for normal operation, but use of the second header "
        "(such as TCP or UDP) is normally considered to be shallow packet inspection (usually called stateful packet "
        "inspection) despite this definition. "
        "There are multiple ways to acquire packets for deep packet inspection. Using port mirroring (sometimes called "
        "Span Port) is a very common way, as well as physically inserting a network tap which duplicates and sends "
        "the data stream to an analyzer tool for inspection. "
        "Deep Packet Inspection (and filtering) enables advanced network management, user service, and security functions "
        "as well as internet data mining, eavesdropping, and internet censorship. Although DPI has been used for Internet "
        "management for many years, some advocates of net neutrality fear that the technique may be used anticompetitively "
        "or to reduce the openness of the Internet. "
        "DPI is used in a wide range of applications, at the so-called 'enterprise' level (corporations and larger institutions), "
        "in telecommunications service providers, and in governments."
    )
    
    # Summarize the text
    summary = summarize(text, top_n=4)
    print("Summary:")
    print(summary)
