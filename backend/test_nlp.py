from nlp_utils import tokenize, compute_nlp_similarities

def test_nlp():
    print("Starting NLP Cosine Similarity Verification...")
    
    jd = "Python software engineer with experience in FastAPI and PostgreSQL database."
    
    # Candidate 1: perfect match
    # Candidate 2: completely irrelevant (frontend React)
    # Candidate 3: strong match, different phrasing
    resumes = [
        "Python developer who writes REST APIs using FastAPI and stores data in PostgreSQL.",
        "Experienced React frontend engineer focused on tailwind css and HTML/Javascript development.",
        "Python backend specialist. Deep knowledge of PostgreSQL, Docker, and FastAPI web APIs."
    ]
    
    scores = compute_nlp_similarities(jd, resumes)
    print(f"Candidate 1 (FastAPI/Postgres): {scores[0]:.4f}")
    print(f"Candidate 2 (React Frontend):    {scores[1]:.4f}")
    print(f"Candidate 3 (Python Backend):    {scores[2]:.4f}")
    
    # Assertions to verify correctness
    assert scores[0] > scores[1], "FastAPI candidate must score higher than React frontend candidate"
    assert scores[2] > scores[1], "Backend candidate must score higher than React frontend candidate"
    assert scores[0] > 0.0, "Similarity should be non-zero"
    
    print("SUCCESS: Mathematical TF-IDF and Cosine Similarity tests passed successfully!")

if __name__ == "__main__":
    test_nlp()
