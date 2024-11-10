from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import re
import numpy as np

def tfidf_sentence_scoring(sentences):
    """Score sentences using TF-IDF."""
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    scores = X.sum(axis=1).A1  # Sum TF-IDF scores for each sentence
    ranked_sentences = [(score, i, sentences[i]) for i, score in enumerate(scores)]
    ranked_sentences.sort(reverse=True, key=lambda x: x[0])
    return ranked_sentences

# Example usage within the summarize function:
def summarize_with_tfidf(text, top_n=3):
    sentences = sent_tokenize(re.sub(r'•|\n', '. ', text))
    ranked_sentences = tfidf_sentence_scoring(sentences)
    top_sentences = sorted([sent for _, _, sent in ranked_sentences[:top_n]])
    return ' '.join(top_sentences)

text = r'''
India, officially the Republic of India,[j][21] is a country in South Asia. It is the seventh-largest country in the world by area and the most populous country. Bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast, it shares land borders with Pakistan to the west;[k] China, Nepal, and Bhutan to the north; and Bangladesh and Myanmar to the east. In the Indian Ocean, India is in the vicinity of Sri Lanka and the Maldives; its Andaman and Nicobar Islands share a maritime border with Thailand, Myanmar, and Indonesia.

Modern humans arrived on the Indian subcontinent from Africa no later than 55,000 years ago.[23][24][25] Their long occupation, initially in varying forms of isolation as hunter-gatherers, has made the region highly diverse, second only to Africa in human genetic diversity.[26] Settled life emerged on the subcontinent in the western margins of the Indus river basin 9,000 years ago, evolving gradually into the Indus Valley Civilisation of the third millennium BCE.[27] By 1200 BCE, an archaic form of Sanskrit, an Indo-European language, had diffused into India from the northwest.[28][29] Its evidence today is found in the hymns of the Rigveda. Preserved by an oral tradition that was resolutely vigilant, the Rigveda records the dawning of Hinduism in India.[30] The Dravidian languages of India were supplanted in the northern and western regions.[31] By 400 BCE, stratification and exclusion by caste had emerged within Hinduism,[32] and Buddhism and Jainism had arisen, proclaiming social orders unlinked to heredity.[33] Early political consolidations gave rise to the loose-knit Maurya and Gupta Empires based in the Ganges Basin.[34] Their collective era was suffused with wide-ranging creativity,[35] but also marked by the declining status of women,[36] and the incorporation of untouchability into an organised system of belief.[l][37] In South India, the Middle kingdoms exported Dravidian-languages scripts and religious cultures to the kingdoms of Southeast Asia.[38]

In the early mediaeval era, Christianity, Islam, Judaism, and Zoroastrianism became established on India's southern and western coasts.[39] Muslim armies from Central Asia intermittently overran India's northern plains,[40] eventually founding the Delhi Sultanate and drawing northern India into the cosmopolitan networks of mediaeval Islam.[41] In the 15th century, the Vijayanagara Empire created a long-lasting composite Hindu culture in south India.[42] In the Punjab, Sikhism emerged, rejecting institutionalised religion.[43] The Mughal Empire, in 1526, ushered in two centuries of relative peace,[44] leaving a legacy of luminous architecture.[m][45] Gradually expanding rule of the British East India Company followed, turning India into a colonial economy but also consolidating its sovereignty.[46] British Crown rule began in 1858. The rights promised to Indians were granted slowly,[47][48] but technological changes were introduced, and modern ideas of education and public life took root.[49] A pioneering and influential nationalist movement emerged, which was noted for nonviolent resistance and became the major factor in ending British rule.[50][51] In 1947, the British Indian Empire was partitioned into two independent dominions,[52][53][54][55] a Hindu-majority dominion of India and a Muslim-majority dominion of Pakistan, amid large-scale loss of life and an unprecedented migration.[56]

India has been a federal republic since 1950, governed through a democratic parliamentary system, and has been the world's most populous democracy since the time of its independence in 1947.[57][58][59] It is a pluralistic, multilingual and multi-ethnic society. India's nominal per capita income increased from US$64 annually in 1951 to US$2,601 in 2022, and its literacy rate from 16.6% to 74%. During the same time, its population grew from 361 million to almost 1.4 billion,[60] and India became the most populous country in 2023.[61][62] From being a comparatively destitute country in 1951,[63] India has become a fast-growing major economy and a hub for information technology services, with an expanding middle class.[64] India has a space programme with several planned or completed extraterrestrial missions. Indian movies, music, and spiritual teachings play an increasing role in global culture.[65] India has substantially reduced its rate of poverty, though at the cost of increasing economic inequality.[66] India is a nuclear-weapon state, which ranks high in military expenditure. It has disputes over Kashmir with its neighbours, Pakistan and China, unresolved since the mid-20th century.[67] Among the socio-economic challenges India faces are gender inequality, child malnutrition,[68] and rising levels of air pollution.[69] India's land is megadiverse, with four biodiversity hotspots.[70] Its forest cover comprises 21.7% of its area.[71] India's wildlife, which has traditionally been viewed with tolerance in India's culture,[72] is supported among these forests, and elsewhere, in protected habitats.

Etymology
Main article: Names for India
According to the Oxford English Dictionary (third edition 2009), the name "India" is derived from the Classical Latin India, a reference to South Asia and an uncertain region to its east. In turn the name "India" derived successively from Hellenistic Greek India ( Ἰνδία), ancient Greek Indos ( Ἰνδός), Old Persian Hindush (an eastern province of the Achaemenid Empire), and ultimately its cognate, the Sanskrit Sindhu, or "river", specifically the Indus River and, by implication, its well-settled southern basin.[73][74] The ancient Greeks referred to the Indians as Indoi (Ἰνδοί), which translates as "The people of the Indus".[75]

The term Bharat (Bhārat; pronounced [ˈbʱaːɾət] ⓘ), mentioned in both Indian epic poetry and the Constitution of India,[76][77] is used in its variations by many Indian languages. A modern rendering of the historical name Bharatavarsha, which applied originally to North India,[78][79] Bharat gained increased currency from the mid-19th century as a native name for India.[76][80]

Hindustan ([ɦɪndʊˈstaːn] ⓘ) is a Middle Persian name for India that became popular by the 13th century,[81] and was used widely since the era of the Mughal Empire. The meaning of Hindustan has varied, referring to a region encompassing the northern Indian subcontinent (present-day northern India and Pakistan) or to India in its near entirety
'''

summary = summarize_with_tfidf(text, top_n=2)
print("Improved Summary:")
print(summary)
