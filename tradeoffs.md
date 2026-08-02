# Design Choices, NLP Similarity, and Tradeoffs

This document details the engineering decisions, scoring methodologies, and architectural tradeoffs made while developing the **AI Resume Screening Agent**.

---

## 1. NLP Similarity Method
For the core statistical alignment, we developed a custom, pure-Python **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer** and **Cosine Similarity** engine.

### Why Pure Python?
1. **Zero Binary Overhead**: Standard scientific python libraries (`numpy`, `scikit-learn`, `scipy`) compile native C++ extensions. On Windows environments, installing these packages can frequently fail due to missing MSVC build tools or DLL link mismatches. A pure-python implementation installs instantly and runs on any environment.
2. **Deterministic Baseline**: Cosine similarity mathematically measures the angular distance between document frequency vectors. It is highly reliable at identifying exact keyword overlaps (e.g., "FastAPI", "Terraform", "Figma") which LLMs sometimes miss or gloss over.
3. **Execution Speed**: The TF-IDF computations execute in under 10ms for a batch of 20+ resumes.

### Mathematical Formulation
- **Term Frequency ($TF$):** Measures the relative occurrence of term $t$ in a single document $d$.
  $$TF(t, d) = \frac{\text{Count}(t \text{ in } d)}{\text{Total terms in } d}$$
- **Inverse Document Frequency ($IDF$):** Evaluates how rare or common a word is across the entire corpus $D$ (where the corpus is defined as the Job Description plus all resumes in the batch).
  $$IDF(t) = \ln\left(1 + \frac{|D|}{1 + \text{Count}(d \in D : t \in d)}\right)$$
- **Cosine Similarity:** Computes the dot product of normalized TF-IDF vectors for the Job Description ($A$) and Resume ($B$).
  $$\text{Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

---

## 2. Model Selection & LLM Architecture
We utilized the **OpenAI API** with `gpt-4o-mini` as the semantic evaluator.

### Rationale
- **Cost and Latency**: `gpt-4o-mini` is extremely fast (average response time < 1.5 seconds) and highly cost-effective, which is critical for processing batches of 10+ resumes.
- **Structured JSON Mode**: Using FastAPI and Pydantic, the agent requests the LLM to output a strict JSON payload. `gpt-4o-mini` is highly optimized for JSON Schema compliance, preventing JSON parsing runtime exceptions.

### Safe Fallback Mechanism
If the `OPENAI_API_KEY` is not provided or fails, the application switches to **Classical NLP Similarity Mode**. In this mode:
- The TF-IDF cosine similarity scores are computed and used directly for ranking.
- A local regex-based parser scans the resume to extract contact information, education keywords, and matches technical skills from a dictionary.
- The UI gracefully notifies the user that it is running in keyword-matching fallback mode, making the application fully offline-capable and demo-safe.

---

## 3. Hybrid Scoring Formula
To balance exact match metrics with high-level conceptual reading comprehension, the agent uses a **Hybrid Scoring Model**:
$$\text{Hybrid Score} = 0.3 \times (\text{Scaled TF-IDF Score}) + 0.7 \times (\text{LLM Evaluation Score})$$

### Rationale for Weighting
- **NLP Score (30%)**: Acts as a keyword safety anchor. If a candidate claims to be a Python developer but doesn't mention the word "Python" (or related tooling) in their text, the mathematical similarity penalizes them.
- **LLM Score (70%)**: Assesses qualitative metrics that TF-IDF cannot compute:
  - **Years of Experience**: TF-IDF sees "6 years" and "1 year" as similar term weights. The LLM understands the temporal difference.
  - **Project Context**: The LLM reads description paragraphs and assesses the complexity of the roles held.
  - **Requirement Gaps**: The LLM audits if the candidate lacks core requirements (e.g. "needs AWS but only has Azure").

---

## 4. Tradeoffs and Future Improvements

### Tradeoff 1: Vector Space Synonyms
*   **Limitation**: TF-IDF treats "Golang" and "Go" or "ReactJS" and "React" as completely separate dimensions, resulting in lower scores if matching across synonyms.
*   **Future Fix**: Replace TF-IDF with a local sentence-transformer embedding model (e.g. `all-MiniLM-L6-v2`) or OpenAI Embeddings to compute dense vector cosine similarity.

### Tradeoff 2: Scanned PDFs & OCR
*   **Limitation**: The parser relies on `pypdf` which extracts digital text. Scanned images of resumes (image-only PDFs) will return empty strings.
*   **Future Fix**: Integrate a lightweight OCR package (like `pytesseract` or OCR API) to extract text from images.

### Tradeoff 3: Rate Limits & Performance
*   **Limitation**: Processing 30 resumes simultaneously results in 30 concurrent API calls to OpenAI, which could throttle rate limits.
*   **Future Fix**:
    1. Implement an async semaphore batching pool (e.g., maximum 5 concurrent LLM calls).
    2. Use the TF-IDF score to filter the top 15 candidates first, and perform the expensive LLM analysis only on that ranked subset.
