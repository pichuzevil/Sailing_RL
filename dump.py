import os

def flatten_repo(root_dir, output_file, extensions=('.py', '.yaml', '.txt')):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(root_dir):
            # Ignore hidden folders and common heavy directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', '__pycache__', 'data']]
            
            for file in files:
                if file.endswith(extensions):
                    file_path = os.path.join(root, file)
                    outfile.write(f"\n{'='*20}\nFILE: {file_path}\n{'='*20}\n")
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                        outfile.write(infile.read())
                    outfile.write("\n")

flatten_repo('.', 'full_project_dump.txt')