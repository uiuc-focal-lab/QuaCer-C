import os

def combine_python_files(source_dir, destination_file):
    """
    Combines the contents of all Python files (.py extension) in a source directory
    into a single text file.

    Args:
        source_dir: The path to the directory containing the Python files.
        destination_file: The path to the text file where the combined content will be written.
    """

    try:
        with open(destination_file, 'w', encoding='utf-8') as outfile:  # Open in write mode, handle encoding
            for filename in os.listdir(source_dir):
                source_path = os.path.join(source_dir, filename)

                if os.path.isfile(source_path) and filename.endswith(".py"):
                    try:
                        with open(source_path, 'r', encoding='utf-8') as infile:  # Open in read mode, handle encoding
                            outfile.write(f"### Start of file: {filename} ###\n") # File separator
                            outfile.write(infile.read())
                            outfile.write(f"\n### End of file: {filename} ###\n\n") # File separator
                        print(f"Added '{filename}' to '{destination_file}'")
                    except UnicodeDecodeError as e: # Handle UnicodeDecodeError if files have different encodings
                        print(f"Error reading '{filename}': {e}. Skipping this file.")

                    except IOError as e:
                        print(f"Unable to read '{filename}': {e}")

    except IOError as e:
        print(f"Unable to write to '{destination_file}': {e}")



if __name__ == "__main__":
    source_directory = input("Enter the source directory: ")
    destination_file_path = input("Enter the destination file path: ")

    combine_python_files(source_directory, destination_file_path)
    print("File combination complete.")