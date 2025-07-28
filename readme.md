# Persona-Based PDF Analysis and Summarization

This project provides an intelligent PDF analysis pipeline that extracts, ranks, and summarizes relevant sections of PDF documents based on a given **persona** and their **job-to-be-done** (JTBD). It is ideal for creating contextual, role-based summaries to aid decision-making or planning tasks.

---

## Overview (Approach)

The solution follows a structured pipeline:

1. **Input Creation**:

   * Scans a directory for PDF files.
   * Prompts the user to define a **Persona Role** and a **Job-to-be-done Task**.
   * Creates an input JSON with metadata.

2. **PDF Parsing and Markdown Extraction**:

   * Uses `PyMuPDF` to extract content with formatting.
   * Converts pages into a structured markdown-like format.
   * Segments documents into titled sections.

3. **Relevance Ranking**:
   * Embedding Model: `paraphrase-MiniLM-L6-v2`
   * Embeds both the persona-task query and document sections using `sentence-transformers`.
   * Ranks sections based on cosine similarity.

4. **Summarization**:

   * Uses a HuggingFace summarization model (`Falconai/text-summerization`) to generate concise summaries.
   * Fallback strategies ensure robustness in case of model errors or short text.

5. **Output**:

   * Outputs a JSON with:

     * Metadata
     * Top-ranked sections
     * Refined summaries
     * Source PDF and page numbers

## Models and Libraries Used

| Component         | Model/Library                                                   |
| ----------------- | --------------------------------------------------------------- |
| PDF Parsing       | `PyMuPDF (fitz)`                                                |
| Text Embedding    | `sentence-transformers: paraphrase-MiniLM-L6-v2`                |
| Summarization     | `transformers: Falconsai/text_summarization`                    |
| Vector Similarity | `cosine similarity (util.cos_sim)` from `sentence-transformers` |
| Utility Libraries | `json`, `os`, `sys`, `re`, `datetime`, `time`                   |

---

## Using Docker

### 1. Build the Docker Image

Make sure your terminal is in the root directory (where the Dockerfile exists):

```bash
docker build -t sol2:id2 . 
```

### 2. Run the Container

You need to mount the folder containing your PDF files to the container.

```bash
docker run --rm -it --network none \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/model:/app/model \
  -v $(pwd):/app \
  sol2:id2
```
---

## Notes:

* The script will prompt for Persona Role and Job-to-be-Done via command line.
* Output (`challenge1b_output.json`) will be saved in the host machine with above command.
* Important: Before running the docker build command please download the model files from [here](https://drive.google.com/drive/folders/18MOpw__TyYojenJfQoDVYQnzvzqTYcM6?usp=sharing). You'd need to download and extract the files and once you go inside the unzipped folder, you'll get a folder with name `model`. Then proceed with the Docker setup. Copy and save it in the project root. Using this since our Git LFS is already used up.

---

