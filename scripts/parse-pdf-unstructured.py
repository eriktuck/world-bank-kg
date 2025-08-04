from io import BytesIO
import os

import requests
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json
from unstructured.chunking.title import chunk_by_title


def download_pdf(url):
    """Download PDF from a URL and return it as bytes."""
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download {url}")
    return BytesIO(response.content)

def group_into_windows(chunks, window_size=3, overlap=1):
    """Group chunks into sliding windows with optional overlap."""
    grouped = []
    i = 0
    while i < len(chunks):
        group = chunks[i:i + window_size]
        grouped.append("\n\n".join(group))
        i += window_size - overlap
    return grouped

def save_to_markdown(grouped_chunks, output_path):
    """Save grouped chunks to a markdown file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, group in enumerate(grouped_chunks, 1):
            f.write(f"## Chunk {i}\n\n{group}\n\n---\n\n")

pdf_url = "https://documents1.worldbank.org/curated/en/995361468056343658/pdf/529200PAD0P0801for0disclosure0final.pdf"
base_output_name = "unstructured_output"

def main():
    pdf_bytes = download_pdf(pdf_url)
    elements = partition_pdf(file=pdf_bytes, strategy="hi_res")

    elements_to_json(elements=elements, filename=f"{base_output_name}.json")

    chunks = chunk_by_title(elements)

    with open({base_output_name}.md, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, 1):
            f.write(f"## Chunk {i}\n\n{chunk}\n\n---\n\n")

    # chunks = [
    #     str(el).strip()
    #     for el in elements
    #     if el.category in {"NarrativeText", "Title", "List", "Table"}
    # ]

    # grouped_chunks = group_into_windows(chunks, window_size=3, overlap=1)

    # # Save to file
    # save_to_markdown(grouped_chunks, base_output_name)

    # print(f"\n✅ Output written to {os.path.abspath(base_output_name)}")
    
if __name__ == "__main__":
    main()

    

    