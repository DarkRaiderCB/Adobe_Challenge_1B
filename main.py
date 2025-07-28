import json
import os
import sys
import re
from datetime import datetime
import time
import fitz
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from input_creator import create_challenge_input_from_pdfs

def convert_pdf_to_markdown(page):
    blocks = page.get_text("dict")["blocks"]
    markdown_lines = []
    
    for block in blocks:
        if "lines" in block:
            for line in block["lines"]:
                line_text = ""
                line_formatting = []
                
                for span in line["spans"]:
                    text = span["text"]
                    flags = span["flags"]
                    
                    is_bold = bool(flags & 2**4)
                    is_italic = bool(flags & 2**3)
                    is_underline = bool(flags & 2**1)
                    
                    if text.strip():
                        formatted_text = text
                        if is_bold and is_underline:
                            formatted_text = f"***{text.strip()}***"
                        elif is_bold:
                            formatted_text = f"**{text.strip()}**"
                        elif is_italic:
                            formatted_text = f"*{text.strip()}*"
                        elif is_underline:
                            formatted_text = f"_{text.strip()}_"
                        
                        line_formatting.append({
                            'text': formatted_text,
                            'is_bold': is_bold,
                            'is_underline': is_underline,
                            'is_bold_underline': is_bold and is_underline
                        })
                
                if line_formatting:
                    combined_text = " ".join([item['text'] for item in line_formatting])
                    if combined_text.strip():
                        has_bold_underline = any(item['is_bold_underline'] for item in line_formatting)
                        has_bold = any(item['is_bold'] for item in line_formatting)
                        
                        markdown_lines.append({
                            'text': combined_text.strip(),
                            'has_bold_underline': has_bold_underline,
                            'has_bold': has_bold
                        })
    
    return markdown_lines

def extract_sections_from_markdown(markdown_lines):
    sections = []
    current_section = {
        'title': '',
        'content': []
    }
    
    for line_data in markdown_lines:
        text = line_data['text']
        
        if line_data['has_bold_underline']:
            if current_section['title'] and current_section['content']:
                sections.append({
                    'title': current_section['title'],
                    'content': '\n'.join(current_section['content'])
                })
            
            current_section = {
                'title': text,
                'content': []
            }
        
        elif line_data['has_bold'] and not current_section['title']:
            if current_section['content']:
                sections.append({
                    'title': 'Introduction',
                    'content': '\n'.join(current_section['content'])
                })
            
            current_section = {
                'title': text,
                'content': []
            }
        
        elif line_data['has_bold'] and len(text.split()) <= 10 and not text.endswith('.'):
            if current_section['title'] and current_section['content']:
                sections.append({
                    'title': current_section['title'],
                    'content': '\n'.join(current_section['content'])
                })
            
            current_section = {
                'title': text,
                'content': []
            }
        
        else:
            if text and not (text.startswith('**') and text.endswith('**')):
                current_section['content'].append(text)
    
    if current_section['title'] and current_section['content']:
        sections.append({
            'title': current_section['title'],
            'content': '\n'.join(current_section['content'])
        })
    elif current_section['content']:
        sections.append({
            'title': 'Document Content',
            'content': '\n'.join(current_section['content'])
        })
    
    return sections

def clean_markdown_text(text):
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'\1', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_pdfs_to_sections(pdf_directory, document_filenames):
    all_sections = []
    print("-> Parsing PDF documents using markdown conversion...", file=sys.stderr)
    
    for filename in document_filenames:
        full_path = os.path.join(pdf_directory, filename)
        if not os.path.exists(full_path):
            print(f"Warning: PDF file not found at {full_path}. Skipping.", file=sys.stderr)
            continue

        try:
            doc = fitz.open(full_path)
            for page_num, page in enumerate(doc):
                markdown_lines = convert_pdf_to_markdown(page)
                
                if markdown_lines:
                    sections = extract_sections_from_markdown(markdown_lines)
                    
                    for section in sections:
                        clean_title = clean_markdown_text(section['title'])
                        clean_content = clean_markdown_text(section['content'])
                        
                        if clean_content:
                            all_sections.append({
                                "document": filename,
                                "page_number": page_num + 1,
                                "section_title": clean_title,
                                "content": clean_content,
                                "raw_markdown_title": section['title'],
                                "raw_markdown_content": section['content']
                            })
                else:
                    text = page.get_text("text")
                    if text.strip():
                        first_line = next((line for line in text.strip().split('\n') if line.strip()), "Document Content")
                        all_sections.append({
                            "document": filename,
                            "page_number": page_num + 1,
                            "section_title": first_line,
                            "content": text.strip(),
                            "raw_markdown_title": first_line,
                            "raw_markdown_content": text.strip()
                        })
            doc.close()
        except Exception as e:
            print(f"Warning: Could not process {filename}. Error: {e}. Skipping.", file=sys.stderr)

    print(f"-> Finished parsing. Found {len(all_sections)} total sections.", file=sys.stderr)
    return all_sections

def generate_summary(text, summarizer, max_words=100, persona="", job_to_be_done=""):
    """
    Generate a proper summary using the summarization model.
    Updated to ensure consistent result fetching pattern from language model.
    """
    try:
        # Clean the input text
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        cleaned_text = re.sub(r'[^\w\s.,!?;:()\-]', '', cleaned_text)  # Remove special chars
        
        # Don't return early for short text - always attempt summarization
        words = cleaned_text.split()
        
        # Ensure minimum length for summarization
        if len(words) < 10:
            return cleaned_text  # Only return original if extremely short
        
        # Truncate if too long (most models have token limits)
        if len(words) > 800:
            cleaned_text = ' '.join(words[:800])
        
        # Create context-aware prompt for summarization
        if persona and job_to_be_done:
            prompt = f"Summarize this text for a {persona} who needs to {job_to_be_done}. Focus on relevant key points and actionable information: {cleaned_text}"
        elif persona:
            prompt = f"Summarize this text from the perspective of a {persona}, highlighting the most important information: {cleaned_text}"
        elif job_to_be_done:
            prompt = f"Summarize this text focusing on information that helps with {job_to_be_done}: {cleaned_text}"
        else:
            prompt = f"Create a concise summary of the following text, capturing the main points and key information: {cleaned_text}"
        
        # Calculate dynamic lengths based on input
        input_word_count = len(cleaned_text.split())
        
        # Set max_length more aggressively to force summarization
        max_length = min(max_words, max(30, int(input_word_count * 0.4)))  # Reduced ratio
        min_length = min(20, max_length - 10)
        
        print(f"-> Generating summary: input_words={input_word_count}, max_length={max_length}, min_length={min_length}", file=sys.stderr)
        
        # Call the summarization model - returns list with dict containing 'summary_text'
        summary_result = summarizer(
            prompt,
            max_length=max_length,
            min_length=min_length,
            do_sample=True,
            truncation=True
        )
        
        # Consistent result fetching pattern - access first element and 'summary_text' key
        if isinstance(summary_result, list) and len(summary_result) > 0:
            if isinstance(summary_result[0], dict) and 'summary_text' in summary_result[0]:
                summary = summary_result[0]['summary_text']
            else:
                print("-> Warning: Unexpected summarizer output format. Using fallback.", file=sys.stderr)
                summary = str(summary_result[0]) if summary_result else cleaned_text[:max_words]
        else:
            print("-> Warning: Summarizer returned empty result. Using fallback.", file=sys.stderr)
            summary = cleaned_text[:max_words]
        
        # Clean up the summary
        summary = summary.strip()
        
        # Remove any repetition of the prompt
        if summary.lower().startswith(('summarize', 'summary', 'create a')):
            # Try to extract just the actual summary part
            sentences = summary.split('. ')
            if len(sentences) > 1:
                summary = '. '.join(sentences[1:])
        
        # Final word count check
        summary_words = summary.split()
        if len(summary_words) > max_words:
            summary = ' '.join(summary_words[:max_words])
            if not summary.endswith('.'):
                summary += '.'
        
        # Verify we actually got a summary (not just the original text)
        if len(summary.split()) >= len(cleaned_text.split()) * 0.8:
            print("-> Warning: Summary is too similar to original. Forcing shorter summary.", file=sys.stderr)
            # Force a much shorter summary
            shorter_result = summarizer(
                f"Write a brief 2-3 sentence summary: {cleaned_text[:500]}",
                max_length=50,
                min_length=15,
                do_sample=True,
                truncation=True
            )
            
            # Consistent result fetching for shorter summary too
            if isinstance(shorter_result, list) and len(shorter_result) > 0:
                if isinstance(shorter_result[0], dict) and 'summary_text' in shorter_result[0]:
                    summary = shorter_result[0]['summary_text']
                else:
                    summary = str(shorter_result[0]) if shorter_result else summary
        
        print(f"-> Summary generated successfully: {len(summary.split())} words", file=sys.stderr)
        return summary
        
    except Exception as e:
        print(f"Warning: Error generating summary: {e}. Creating manual summary.", file=sys.stderr)
        
        # Manual summarization as fallback
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) <= 3:
            return cleaned_text
        
        # Take first, middle, and last sentences for a basic summary
        summary_sentences = [sentences[0]]
        if len(sentences) > 2:
            summary_sentences.append(sentences[len(sentences)//2])
        if len(sentences) > 1:
            summary_sentences.append(sentences[-1])
        
        manual_summary = ' '.join(summary_sentences)
        words = manual_summary.split()
        if len(words) > max_words:
            manual_summary = ' '.join(words[:max_words]) + '.'
        
        return manual_summary

def extract_key_sentences(content, query_embedding, model, num_sentences=3):
    sentences = re.split(r'(?<=[.!?])\s+', content.replace('\n', ' '))
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    
    if not sentences:
        return content[:200] + "..." if len(content) > 200 else content
    
    if len(sentences) <= num_sentences:
        return " ".join(sentences)
    
    try:
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        sentence_scores = util.cos_sim(query_embedding, sentence_embeddings)
        
        scored_sentences = list(zip(sentences, sentence_scores[0]))
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        top_sentences = [s[0] for s in scored_sentences[:num_sentences]]
        
        original_order_sentences = []
        for sentence in sentences:
            if sentence in top_sentences:
                original_order_sentences.append(sentence)
                if len(original_order_sentences) == num_sentences:
                    break
        
        return " ".join(original_order_sentences)
    except Exception as e:
        print(f"Warning: Error in sentence extraction: {e}. Using first few sentences.", file=sys.stderr)
        return " ".join(sentences[:num_sentences])

def run_persona_analysis(input_json_path, pdf_directory):
    start_time = time.time()
    print(f"-> Code execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)
    
    with open(input_json_path, 'r') as f:
        input_data = json.load(f)

    persona = input_data.get("persona", {}).get("role", "")
    job_to_be_done = input_data.get("job_to_be_done", {}).get("task", "")
    document_filenames = [doc.get("filename") for doc in input_data.get("documents", [])]
    
    query = f"{persona}: {job_to_be_done}"
    print(f"-> Starting analysis for query: {query}\n", file=sys.stderr)

    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    
    print("-> Loading sentence-transformer model (paraphrase-MiniLM-L6-v2)...", file=sys.stderr)
    try:
        sentence_model_path = os.path.join(model_dir, 'paraphrase-MiniLM-L6-v2')
        model = SentenceTransformer(sentence_model_path)
        print("-> Sentence transformer model loaded successfully.", file=sys.stderr)
    except Exception as e:
        print(f"Error loading sentence transformer model: {e}", file=sys.stderr)
        return

    print("-> Loading summarization model (Falconsai/text_summarization)...", file=sys.stderr)
    try:
        summarizer_model_path = os.path.join(model_dir, 'text_summarization')
        summarizer = pipeline(
            "summarization",
            model=summarizer_model_path,
            tokenizer=summarizer_model_path,
            device=-1
        )
        print("-> Summarization model loaded successfully.", file=sys.stderr)
    except Exception as e:
        print(f"Error loading summarization model: {e}", file=sys.stderr)
        return

    sections = parse_pdfs_to_sections(pdf_directory, document_filenames)
    if not sections:
        print("Error: No sections were extracted from PDFs. Cannot proceed.", file=sys.stderr)
        return

    print("-> Embedding query and document sections...", file=sys.stderr)
    try:
        query_embedding = model.encode(query, convert_to_tensor=True)
        section_contents = [sec["content"] for sec in sections]
        section_embeddings = model.encode(section_contents, convert_to_tensor=True)
        print("-> Embedding complete.", file=sys.stderr)
    except Exception as e:
        print(f"Error during embedding: {e}", file=sys.stderr)
        return

    print("-> Calculating relevance scores for section ranking...", file=sys.stderr)
    try:
        cosine_scores = util.cos_sim(query_embedding, section_embeddings)

        for i, sec in enumerate(sections):
            sec['relevance_score'] = cosine_scores[0][i].item()
        
        ranked_sections = sorted(sections, key=lambda x: x['relevance_score'], reverse=True)
    except Exception as e:
        print(f"Error calculating relevance scores: {e}", file=sys.stderr)
        return

    print("-> Performing sub-section analysis and summary generation on top sections...", file=sys.stderr)
    subsection_analysis_results = []
    top_n_sections = ranked_sections[:5]

    for i, section in enumerate(top_n_sections):
        print(f"-> Processing section {i+1}/{len(top_n_sections)}: {section['section_title'][:50]}...", file=sys.stderr)
        print(f"-> Original content length: {len(section['content'].split())} words", file=sys.stderr)
        
        try:
            summary = generate_summary(
                section['content'], 
                summarizer, 
                max_words=100,
                persona=persona,
                job_to_be_done=job_to_be_done
            )
            
            print(f"-> Generated summary length: {len(summary.split())} words", file=sys.stderr)
            
            clean_original_title = clean_markdown_text(section.get("raw_markdown_title", section["section_title"]))
            
            subsection_analysis_results.append({
                "document": section["document"],
                "refined_text": summary,
                "page_number": section["page_number"]
            })
        except Exception as e:
            print(f"Warning: Error processing section '{section['section_title']}': {e}", file=sys.stderr)
            
            # Even for fallback, try to create a shorter version
            fallback_text = section["content"]
            words = fallback_text.split()
            if len(words) > 100:
                fallback_text = ' '.join(words[:100]) + "..."
            
            clean_original_title = clean_markdown_text(section.get("raw_markdown_title", section["section_title"]))
            
            subsection_analysis_results.append({
                "document": section["document"],  
                "refined_text": fallback_text,
                "page_number": section["page_number"]
            })

    print("-> Formatting final output JSON...", file=sys.stderr)
    final_output = {
        "metadata": {
            "input_documents": document_filenames,
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [
            {
                "document": sec["document"],
                "section_title": clean_markdown_text(sec.get("raw_markdown_title", sec["section_title"])),
                "importance_rank": i + 1,
                "page_number": sec["page_number"]
            } for i, sec in enumerate(top_n_sections)
        ],
        "subsection_analysis": subsection_analysis_results
    }

    output_filename = "challenge1b_output.json"
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        
        print(f"\nAnalysis complete. Output saved to {output_filename}", file=sys.stderr)
        print(f"Found {len(sections)} sections total, analyzed top {len(top_n_sections)} sections.", file=sys.stderr)
        print(f"Generated proper summaries for each analyzed section.", file=sys.stderr)
        
        end_time = time.time()
        total_execution_time = end_time - start_time
        print(f"-> Code execution ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)
        print(f"-> Total execution time: {total_execution_time:.2f} seconds", file=sys.stderr)
    except Exception as e:
        print(f"Error saving output file: {e}", file=sys.stderr)

if __name__ == '__main__':
    pdf_path, input_json = create_challenge_input_from_pdfs("input")
    input_json_path = input_json
    pdf_directory_path = pdf_path

    if not os.path.exists(input_json_path):
        print(f"Error: Input JSON file not found at {input_json_path}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(pdf_directory_path):
        print(f"Error: PDF directory not found at {pdf_directory_path}", file=sys.stderr)
        sys.exit(1)
    
    run_persona_analysis(input_json_path, pdf_directory_path)
