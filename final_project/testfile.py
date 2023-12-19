import spacy
from sklearn.metrics.pairwise import cosine_similarity
from nltk import ngrams
from statistics import mean
from textstat import flesch_kincaid_grade, dale_chall_readability_score

# Load Word Embedding models
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 1500000


def sentence_embedding(sentence):
    # Sentence embeddings
    doc = nlp(sentence)
    return doc.vector.reshape(1, -1)

def jaccard_similarity(sentence1, sentence2, n=1):
    # Calculate Jaccard similarity
    ngrams1 = set(ngrams(sentence1.split(), n))
    ngrams2 = set(ngrams(sentence2.split(), n))
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    return len(intersection) / len(union)

def calculate_cosine_similarity(sentence1, sentence2):
    # Calculate cosine similarity
    vector1 = sentence_embedding(sentence1)
    vector2 = sentence_embedding(sentence2)
    similarity = cosine_similarity(vector1, vector2)[0][0]
    return similarity

def evaluate_english_simplification(original_text, simplified_text):
    # Tokenize sentences
    original_sentences = [sentence.text for sentence in nlp(original_text).sents]
    simplified_sentences = [sentence.text for sentence in nlp(simplified_text).sents]

    # Calculate similarity scores
    similarity_scores = [calculate_cosine_similarity(sent1, sent2) for sent1, sent2 in zip(original_sentences, simplified_sentences)]
    jaccard_scores = [jaccard_similarity(sent1, sent2) for sent1, sent2 in zip(original_sentences, simplified_sentences)]

    # Dale–Chall Readability Score
    dale_chall_simplified = dale_chall_readability_score(simplified_text)
    dale_chall_original= dale_chall_readability_score(original_text)


    # Flesch-Kincaid Readability Grade Level
    flesch_kincaid_simplified = flesch_kincaid_grade(simplified_text)
    flesch_kincaid_original= flesch_kincaid_grade(original_text)


    similarity_score = mean(similarity_scores)
    jaccard_score = mean(jaccard_scores)


    return {
        "Similarity Score": similarity_score,
        "Jaccard Score": jaccard_score,
        "Dale–Chall Score": dale_chall_simplified,
        "Dale–Chall Score Original Text": dale_chall_original,
        "Flesch-Kincaid Score": flesch_kincaid_simplified,
        "Flesch-Kincaid Score Original Text": flesch_kincaid_original,
    }

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

# Example file paths
original_file_path = "normal_test.txt"
simplified_file_path = "normal_test.txt"

# Read content from files
original_text = read_text_file(original_file_path)
simplified_text = read_text_file(simplified_file_path)

# Evaluate the simplified text
evaluation_scores = evaluate_english_simplification(original_text, simplified_text)

# Print the individual scores
for criterion, score in evaluation_scores.items():
    print(f"{criterion}: {score}")
