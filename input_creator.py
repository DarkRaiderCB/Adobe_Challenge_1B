import json
import os
import sys

def create_challenge_input_from_pdfs(pdf_directory, output_filename="challenge1b_input.json"):
    try:
        all_files = os.listdir(pdf_directory)
        pdf_files = sorted([f for f in all_files if f.lower().endswith('.pdf')])
        if not pdf_files:
            print(f"Error: No PDF files found in the directory: {pdf_directory}", file=sys.stderr)
            return
    except FileNotFoundError:
        print(f"Error: The specified directory '{pdf_directory}' does not exist.", file=sys.stderr)
        print("Please provide a valid path containing PDF files.", file=sys.stderr)
        return

    documents_list = []
    for pdf_filename in pdf_files:
        base_name = os.path.splitext(pdf_filename)[0]
        
        documents_list.append({
            "filename": pdf_filename,
            "title": base_name
        })

    persona_role = input("Enter the Persona Role (e.g., Travel Planner): ")
    job_task = input("Enter the Job-to-be-done Task (e.g., Plan a trip...): ")

    challenge_data = {
        "challenge_info": {
            "challenge_id": "round_1b",
            "test_case_name": "custom_analysis",
            "description": "Analysis based on Round 1A outputs"
        },
        "documents": documents_list,
        "persona": {
            "role": persona_role
        },
        "job_to_be_done": {
            "task": job_task
        }
    }

    with open(output_filename, 'w') as f:
        json.dump(challenge_data, f, indent=4)
    
    print(f"Successfully created {output_filename}", file=sys.stderr)

    # return pdf directory path for further processing if needed, and also input json formed
    return pdf_directory, output_filename


# if __name__ == '__main__':
#     pdf_input_directory = "input"  # Replace with your folder name or make dynamic via input()
#     create_challenge_input_from_pdfs(pdf_input_directory)
