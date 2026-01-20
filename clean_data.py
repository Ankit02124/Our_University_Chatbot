import os
import re

# Paths
SOURCE_DIR = "data"
CLEAN_DIR = "data_clean"

def clean_text(text):
    """Removes source tags, page numbers, and extra noise."""
    # Remove tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove lines that are just numbers (page numbers)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def main():
    if not os.path.exists(CLEAN_DIR):
        os.makedirs(CLEAN_DIR)

    for filename in os.listdir(SOURCE_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(SOURCE_DIR, filename)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                cleaned_content = clean_text(content)
                
                clean_path = os.path.join(CLEAN_DIR, filename)
                with open(clean_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_content)
                
                print(f"✅ Successfully cleaned: {filename}")
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

if __name__ == "__main__":
    main()