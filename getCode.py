# extract_code.py
"""
This script searches for all Python files (.py) in its current directory
and all subdirectories. It then reads the content of each file and
combines them into a single, large text file.

The output file will be named 'all_code_combined.txt' and will be created
in the same directory where the script is run.

Each file's content in the output is preceded by a header that indicates
the original file's path, making it easy to navigate the combined code.

Usage:
1. Save this code as a Python file (e.g., `extract_code.py`).
2. Place it in the root directory of the project you want to extract code from.
3. Run it from your terminal: python extract_code.py
"""
import os

# --- Configuration ---
# The name of the output file that will contain all the code.
OUTPUT_FILENAME = "all_code_combined.txt"

# The directory to start searching from. "." means the current directory.
START_DIRECTORY = "."

# A list of directories to exclude from the search.
# Useful for ignoring virtual environments, build artifacts, etc.
EXCLUDE_DIRS = {"__pycache__", ".venv", "venv", ".git", "env"}

# A list of files to exclude from the search.
# We exclude this script itself to avoid including it in the output.
EXCLUDE_FILES = {os.path.basename(__file__), OUTPUT_FILENAME}
# ---

def extract_python_code():
    """
    Walks through directories, finds .py files, and writes their contents
    to a single output file.
    """
    print(f"Starting code extraction from '{os.path.abspath(START_DIRECTORY)}'...")
    
    # Keep track of the number of files processed.
    file_count = 0

    try:
        # Open the output file in write mode ('w').
        # Using 'with' ensures the file is automatically closed.
        # 'encoding='utf-8'' is important for handling a wide range of characters.
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as outfile:
            
            # os.walk is the perfect tool for recursively traversing a directory tree.
            for dirpath, dirnames, filenames in os.walk(START_DIRECTORY):
                
                # Modify dirnames in-place to exclude specified directories from the walk.
                # This is more efficient than just checking the path later.
                dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
                
                for filename in filenames:
                    # Check if the file is a Python file and not in our exclusion list.
                    if filename.endswith('.py') and filename not in EXCLUDE_FILES:
                        
                        # Construct the full, relative path to the file.
                        # This looks cleaner than an absolute path in the output.
                        relative_path = os.path.join(dirpath, filename)
                        
                        print(f"  -> Processing: {relative_path}")
                        
                        # Write a clear header for each file into the output file.
                        outfile.write("=" * 80 + "\n")
                        outfile.write(f"FILENAME: {relative_path}\n")
                        outfile.write("=" * 80 + "\n\n")
                        
                        try:
                            # Open and read the content of the Python file.
                            with open(relative_path, 'r', encoding='utf-8') as infile:
                                content = infile.read()
                                outfile.write(content)
                        except Exception as e:
                            # If a file can't be read, log the error and continue.
                            error_message = f"!!! ERROR READING FILE: {relative_path} - {e} !!!"
                            print(error_message)
                            outfile.write(error_message)
                        
                        # Add a few newlines to separate files cleanly.
                        outfile.write("\n\n\n")
                        file_count += 1

        print("\n" + "=" * 30)
        print("Extraction complete!")
        print(f"Processed {file_count} Python files.")
        print(f"All code has been combined into: '{os.path.abspath(OUTPUT_FILENAME)}'")
        print("=" * 30)

    except IOError as e:
        print(f"Error: Could not write to output file '{OUTPUT_FILENAME}'. Reason: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# This is standard Python practice to make the script runnable.
if __name__ == "__main__":
    extract_python_code()