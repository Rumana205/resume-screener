import re
import math
from typing import List, Dict, Set

# Standard English stopwords to filter out noisy words
STOPWORDS: Set[str] = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could',
    "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for',
    'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's",
    'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm",
    "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't",
    'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours',
    'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't",
    'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there',
    "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too',
    'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't",
    'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why',
    "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours',
    'yourself', 'yourselves'
}

def tokenize(text: str) -> List[str]:
    """
    Tokenizes text by lowercasing and extracting alphanumeric words.
    """
    if not text:
        return []
    # Lowercase and split on non-alphanumeric boundaries
    words = re.findall(r'\b[a-z0-9\-]+\b', text.lower())
    # Remove stopwords
    return [word for word in words if word not in STOPWORDS]

def compute_tf(tokens: List[str]) -> Dict[str, float]:
    """
    Computes term frequency (TF) for a tokenized document.
    TF = count of term in document / total terms in document.
    """
    if not tokens:
        return {}
    tf = {}
    for token in tokens:
        tf[token] = tf.get(token, 0.0) + 1.0
    
    total_tokens = len(tokens)
    for token in tf:
        tf[token] = tf[token] / total_tokens
    return tf

def compute_idf(documents_tokens: List[List[str]]) -> Dict[str, float]:
    """
    Computes inverse document frequency (IDF) for all terms in a corpus of tokenized documents.
    IDF = ln(1 + total_documents / (1 + doc_frequency_of_term))
    """
    num_docs = len(documents_tokens)
    if num_docs == 0:
        return {}
        
    doc_freq = {}
    for doc in documents_tokens:
        unique_terms = set(doc)
        for term in unique_terms:
            doc_freq[term] = doc_freq.get(term, 0) + 1
            
    idf = {}
    for term, freq in doc_freq.items():
        # Smoothed IDF formula
        idf[term] = math.log(1.0 + (num_docs / (1.0 + freq)))
    return idf

def get_tfidf_vector(tf: Dict[str, float], idf: Dict[str, float], vocabulary: Set[str]) -> Dict[str, float]:
    """
    Builds a sparse TF-IDF vector mapping word -> tf-idf value for the given vocabulary.
    """
    vector = {}
    for word in vocabulary:
        word_tf = tf.get(word, 0.0)
        word_idf = idf.get(word, 0.0)
        vector[word] = word_tf * word_idf
    return vector

def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """
    Computes the cosine similarity between two sparse TF-IDF vectors.
    """
    # Find all common features in the two vectors
    common_words = set(vec_a.keys()).union(set(vec_b.keys()))
    
    dot_product = 0.0
    sum_sq_a = 0.0
    sum_sq_b = 0.0
    
    for word in common_words:
        val_a = vec_a.get(word, 0.0)
        val_b = vec_b.get(word, 0.0)
        
        dot_product += val_a * val_b
        sum_sq_a += val_a * val_a
        sum_sq_b += val_b * val_b
        
    norm_a = math.sqrt(sum_sq_a)
    norm_b = math.sqrt(sum_sq_b)
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
        
    return dot_product / (norm_a * norm_b)

def compute_nlp_similarities(job_description: str, resumes: List[str]) -> List[float]:
    """
    Helper function to compute cosine similarity scores for a list of resumes against a job description.
    Returns a list of similarity scores (floats between 0.0 and 1.0) in the same order as resumes.
    """
    jd_tokens = tokenize(job_description)
    resumes_tokens = [tokenize(res) for res in resumes]
    
    # Vocabulary contains all unique words across the job description and all resumes
    vocabulary = set(jd_tokens)
    for res_tok in resumes_tokens:
        vocabulary.update(res_tok)
        
    # Corpus includes the job description itself + all resumes
    corpus = [jd_tokens] + resumes_tokens
    
    # Compute IDFs over the corpus
    idf = compute_idf(corpus)
    
    # Vectorize Job Description
    jd_tf = compute_tf(jd_tokens)
    jd_vector = get_tfidf_vector(jd_tf, idf, vocabulary)
    
    # Vectorize each resume and compute similarity
    scores = []
    for res_tokens in resumes_tokens:
        res_tf = compute_tf(res_tokens)
        res_vector = get_tfidf_vector(res_tf, idf, vocabulary)
        
        sim = cosine_similarity(jd_vector, res_vector)
        scores.append(round(sim, 4))
        
    return scores
