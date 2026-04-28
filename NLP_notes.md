# Complete NLP (Natural Language Processing) from A to Z

> **Beginner-friendly notes with theory, code, use cases, interview points, and practical explanations**

---

# Table of Contents

* What is NLP?
* NLP Pipeline
* Libraries and Pretrained Models
* Regex
* Tokenization
* Stopwords
* Stemming vs Lemmatization
* POS Tagging
* Named Entity Recognition (NER)
* N-grams
* Bag of Words (BOW)
* TF-IDF
* Word Embeddings
* Word2Vec, GloVe, fastText
* Sentence Embeddings
* Text Classification (Spam Detection)
* Transformers (BERT, GPT)
* spaCy vs NLTK
* Interview Questions

---

# What is NLP?

## Full Form

**NLP = Natural Language Processing**

## Definition

NLP is a branch of **Artificial Intelligence (AI)** that helps computers understand, process, analyze, and generate human language.

Humans understand:

* Text
* Speech
* Meaning
* Context

NLP tries to make machines do the same.

---

## Real-Life Applications

* Gmail Spam Filter
* ChatGPT
* Google Translate
* Siri / Alexa
* Sentiment Analysis
* Resume Screening
* Disease Symptom Chatbots
* Auto-completion
* Customer Support Bots

---

# NLP Pipeline (Big Picture)

```text
Raw Text
↓
Text Cleaning
↓
Tokenization
↓
Stopword Removal
↓
Stemming / Lemmatization
↓
Feature Extraction
↓
Vectorization
↓
Model Building
↓
Prediction
```

---

# Python Libraries Used in NLP

## Common Libraries

1. **NLTK** — Natural Language Toolkit
2. **spaCy** — Production-grade NLP library
3. **Scikit-learn** — Machine learning models
4. **Gensim** — Word embeddings
5. **TensorFlow** — Deep learning
6. **PyTorch** — Deep learning
7. **Hugging Face Transformers** — BERT, GPT, T5 models

---

## Pretrained Models

* **fastText** (Meta/Facebook)
* **TensorFlow Hub** (Google)
* **Sentence-BERT**
* **Word2Vec**
* **GloVe**

---

# Open Source Ecosystem Advantages

Earlier:

* Heavy C++ implementations
* Expensive hardware
* Difficult training

Now:

* Python libraries reduce code drastically
* AWS, Azure, Google Cloud provide affordable compute
* Big tech investment improved model availability

---

# NLP Techniques

## 1. Rules and Heuristics

Examples:

* Regular Expressions
* Rule-based extraction

## 2. Machine Learning

Examples:

* Naive Bayes
* Logistic Regression
* SVM

## 3. Deep Learning / Embeddings

Examples:

* Word2Vec
* Sentence Embeddings
* BERT

---

# Regex (Regular Expressions)

## Definition

Regex identifies patterns inside text.

Examples:

* Phone numbers
* Emails
* Dates
* URLs

## Code

```python
import re

text='Call me 9876543210'
pattern=r'\d{10}'

matches=re.findall(pattern,text)
print(matches)
```

Output:

```python
['9876543210']
```

## Explanation

* `\d` = digit
* `{10}` = repeated 10 times
* `findall()` returns matches

> **Use Case:** Information extraction and text cleaning.

---

# Tokenization

## Definition

Tokenization splits text into smaller pieces called tokens.

Example:

```text
I love NLP
```

becomes:

```text
[I, love, NLP]
```

## Code

```python
from nltk.tokenize import word_tokenize

text='I love NLP'
words=word_tokenize(text)

print(words)
```

## Sentence Tokenization

```python
from nltk.tokenize import sent_tokenize

text='I love NLP. It is amazing.'

print(sent_tokenize(text))
```

## Why needed?

Models work on tokens, not raw paragraphs.

---

# Stopwords

## Definition

Common words often removed:

* the
* is
* and
* of

## Code

```python
from nltk.corpus import stopwords

words=['this','is','good']

filtered=[]

for word in words:
 if word not in stopwords.words('english'):
   filtered.append(word)

print(filtered)
```

Output:

```python
['good']
```

---

# Stemming vs Lemmatization

## Stemming

Cuts word endings.

Examples:

* running → run
* studies → studi

```python
from nltk.stem import PorterStemmer

stemmer=PorterStemmer()
print(stemmer.stem('running'))
```

---

## Lemmatization

Dictionary root.

Examples:

* studies → study
* running → run

```python
import spacy

nlp=spacy.load('en_core_web_sm')

doc=nlp('running studies')

for token in doc:
 print(token.text,token.lemma_)
```

> **Important:** spaCy supports lemmatization but not traditional stemming.

---

# POS Tagging

## Full Form

Part Of Speech Tagging

Assigns:

* Noun
* Verb
* Adjective

```python
doc=nlp('Dog runs fast')

for token in doc:
 print(token.text,token.pos_)
```

Output:

```text
Dog NOUN
runs VERB
fast ADV
```

---

# Named Entity Recognition (NER)

## Finds entities like:

* Person
* Organization
* Location

```python
doc=nlp('Google hired Ravi in Hyderabad')

for ent in doc.ents:
 print(ent.text,ent.label_)
```

Output:

```text
Google ORG
Ravi PERSON
Hyderabad GPE
```

---

# N-Grams

## Types

* Unigram → good
* Bigram → not good
* Trigram → I am happy

```python
from sklearn.feature_extraction.text import CountVectorizer

v=CountVectorizer(ngram_range=(1,2))
```

`(1,2)` means:

* Unigrams
* Bigrams

---

# Text Representation (Vector Space Model)

Representing text as vectors.

Methods:

* One Hot Encoding
* Label Encoding
* Bag of Words
* TF-IDF
* Word Embeddings

> **Important:** Good text representation often matters more than fancy algorithms.

---

# Bag of Words (BOW)

Converts text into numeric counts.

```python
from sklearn.feature_extraction.text import CountVectorizer

vec=CountVectorizer()

X=vec.fit_transform(x_train.values)

print(X.toarray()[:2])
print(len(vec.get_feature_names_out()))
print(vec.vocabulary_)
```

## Important Methods

* `fit()` → learns vocabulary
* `transform()` → converts text
* `fit_transform()` → both

---

## Limitations of BOW / N-grams

* High dimensionality
* Sparsity
* Out-of-vocabulary problem
* Does not capture meaning

---

# TF-IDF

## Full Form

TF = Term Frequency
IDF = Inverse Document Frequency

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vec=TfidfVectorizer()
X=vec.fit_transform(['machine learning is good'])
```

## Why log used in IDF?

Log dampens very large frequency effects.

## Advantages

* Better than raw counts
* Highlights important words

## Limitation

Still does not capture semantics.

---

# Word Embeddings

Convert words into dense vectors.

Properties:

* Similar words have similar vectors
* Lower dimensions
* Capture semantics

Example:

* cat near dog
* king near queen

---

# Word2Vec

```python
import gensim.downloader as api

wv=api.load('word2vec-google-news-300')

print(wv.similarity('king','queen'))
```

Famous relation:

```text
king - man + woman = queen
```

---

# Other Embeddings

## GloVe

Global Vectors

```python
glv=api.load('glove-twitter-25')
```

## fastText

Handles out-of-vocabulary better using subword information.

---

# Sentence Embeddings

Different from word embeddings.

* Word embedding = vector for word
* Sentence embedding = vector for full sentence

Example:
"heart problem"
can match
"cardiac disease"

Models:

* Sentence-BERT
* Universal Sentence Encoder

---

# Cosine Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity
```

Measures similarity using angle between vectors.

Used in:

* Semantic search
* Recommendations
* Document matching

---

# Text Classification Example (Spam Detection)

Pipeline:

```text
Text
↓
CountVectorizer
↓
Naive Bayes
↓
Spam / Not Spam
```

Code:

```python
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

model=Pipeline([
('vectorizer',CountVectorizer()),
('classifier',MultinomialNB())
])
```

---

# Why Embeddings Introduced?

Problem with CountVectorizer:
New similar text may not be understood semantically.

Embeddings solve that.

---

# Transformers

Modern NLP models based on self-attention.

## Self-Attention

Words check which other words matter.

Example:
"The animal didn't cross the street because it was tired"

"it" refers animal.

---

## BERT

**Bidirectional Encoder Representations from Transformers**

Reads context in both directions.

```python
from transformers import pipeline

classifier=pipeline('sentiment-analysis')
print(classifier('I love NLP'))
```

---

## GPT

**Generative Pre-trained Transformer**

Predicts next token to generate text.

---

# Deep Learning Models Before Transformers

* RNN = Recurrent Neural Network
* LSTM = Long Short-Term Memory
* GRU = Gated Recurrent Unit

---

# spaCy vs NLTK

| Feature       | spaCy      | NLTK     |
| ------------- | ---------- | -------- |
| Focus         | Production | Research |
| Stemming      | No         | Yes      |
| Lemmatization | Yes        | Yes      |
| Speed         | Faster     | Slower   |
| Use Case      | Apps       | Learning |

---

# Common Interview Questions

1. Difference between stemming and lemmatization?
2. Difference between BOW and TF-IDF?
3. What is cosine similarity?
4. Why embeddings over count vectors?
5. What is self-attention?
6. Difference between word and sentence embeddings?

---

# Learning Roadmap

Study in order:

1. Regex
2. Tokenization
3. Stopwords
4. Lemmatization
5. POS
6. NER
7. BOW
8. TF-IDF
9. Embeddings
10. Word2Vec
11. Transformers

---

# Practice Projects

* Spam Classifier
* Sentiment Analyzer
* NER Extractor
* Semantic Search
* Chatbot
* Disease Symptom Classifier

---

## Final Note

This covers theory, code, explanations, limitations, use cases, and practical concepts needed for a strong NLP foundation.
