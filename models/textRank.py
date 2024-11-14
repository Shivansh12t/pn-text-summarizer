import re
import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

#we are using networkx for making graphs

def textrank_sentence_scoring(sentences):
    """Score sentences using Google's TextRank algorithm."""
    stop_words = set(stopwords.words('english'))
    vectorizer = CountVectorizer().fit_transform([' '.join([word.lower() for word in word_tokenize(re.sub(r'\W', ' ', s)) if word.lower() not in stop_words]) for s in sentences])
    similarity_matrix = cosine_similarity(vectorizer, vectorizer)

    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((score, i, sentences[i]) for i, score in scores.items()), reverse=True, key=lambda x: x[0])
    return ranked_sentences

def summarize_with_textrank(text, top_n=3):
    sentences = sent_tokenize(re.sub(r'•|\n', '. ', text))
    ranked_sentences = textrank_sentence_scoring(sentences)
    top_sentences = sorted([sent for _, _, sent in ranked_sentences[:top_n]])
    return ' '.join(top_sentences)

# text=r'''
# India, officially the Republic of India,(ISO: Bhārat Gaṇarājya)[20] is a country in South Asia. It is the seventh-largest country in the world by area and the most populous country. Bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast, it shares land borders with Pakistan to the west;[j] China, Nepal, and Bhutan to the north; and Bangladesh and Myanmar to the east. In the Indian Ocean, India is in the vicinity of Sri Lanka and the Maldives; its Andaman and Nicobar Islands share a maritime border with Thailand, Myanmar, and Indonesia.

# Modern humans arrived on the Indian subcontinent from Africa no later than 55,000 years ago.[22][23][24] Their long occupation, initially in varying forms of isolation as hunter-gatherers, has made the region highly diverse, second only to Africa in human genetic diversity.[25] Settled life emerged on the subcontinent in the western margins of the Indus river basin 9,000 years ago, evolving gradually into the Indus Valley Civilisation of the third millennium BCE.[26] By 1200 BCE, an archaic form of Sanskrit, an Indo-European language, had diffused into India from the northwest.[27][28] Its evidence today is found in the hymns of the Rigveda. Preserved by an oral tradition that was resolutely vigilant, the Rigveda records the dawning of Hinduism in India.[29] The Dravidian languages of India were supplanted in the northern and western regions.[30] By 400 BCE, stratification and exclusion by caste had emerged within Hinduism,[31] and Buddhism and Jainism had arisen, proclaiming social orders unlinked to heredity.[32] Early political consolidations gave rise to the loose-knit Maurya and Gupta Empires based in the Ganges Basin.[33] Their collective era was suffused with wide-ranging creativity,[34] but also marked by the declining status of women,[35] and the incorporation of untouchability into an organised system of belief.[k][36] In South India, the Middle kingdoms exported Dravidian-languages scripts and religious cultures to the kingdoms of Southeast Asia.[37]

# In the early mediaeval era, Christianity, Islam, Judaism, and Zoroastrianism became established on India's southern and western coasts.[38] Muslim armies from Central Asia intermittently overran India's northern plains,[39] eventually founding the Delhi Sultanate and drawing northern India into the cosmopolitan networks of mediaeval Islam.[40] In the 15th century, the Vijayanagara Empire created a long-lasting composite Hindu culture in south India.[41] In the Punjab, Sikhism emerged, rejecting institutionalised religion.[42] The Mughal Empire, in 1526, ushered in two centuries of relative peace,[43] leaving a legacy of luminous architecture.[l][44] Gradually expanding rule of the British East India Company followed, turning India into a colonial economy but also consolidating its sovereignty.[45] British Crown rule began in 1858. The rights promised to Indians were granted slowly,[46][47] but technological changes were introduced, and modern ideas of education and public life took root.[48] A pioneering and influential nationalist movement emerged, which was noted for nonviolent resistance and became the major factor in ending British rule.[49][50] In 1947, the British Indian Empire was partitioned into two independent dominions,[51][52][53][54] a Hindu-majority dominion of India and a Muslim-majority dominion of Pakistan, amid large-scale loss of life and an unprecedented migration.[55]
# '''



# summary = summarize_with_textrank(text, top_n=4)
# print("Improved Summary with TextRank:")
# print(summary)
