import os
import requests
from io import BytesIO
from pdfminer.high_level import extract_text
from tqdm import tqdm
from unidecode import unidecode

def download_pdf(url):
    """Download PDF from a URL and return it as bytes."""
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download {url}")
    return BytesIO(response.content)

def extract_text_from_pdf(pdf_bytes):
    """Extract text from a PDF file using pdfminer.six."""
    return extract_text(pdf_bytes)

def clean_text(text):
    """Clean up extracted text for NLP."""
    text = unidecode(text)  # normalize unicode accents
    text = text.replace('\n', ' ')
    text = ' '.join(text.split())  # collapse extra whitespace
    return text

def prepare_document(url):
    """Pipeline to fetch, extract, and clean text from a PDF URL."""
    try:
        pdf_bytes = download_pdf(url)
        raw_text = extract_text_from_pdf(pdf_bytes)
        cleaned_text = clean_text(raw_text)
        return cleaned_text
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

def save_text(text, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

url = "https://documents1.worldbank.org/curated/en/995361468056343658/pdf/529200PAD0P0801for0disclosure0final.pdf"
text = prepare_document(url)

print(text[:1000])

save_text(text, "output.txt")


