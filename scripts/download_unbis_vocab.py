import requests
from bs4 import BeautifulSoup
import json

def get_unbis_vocab():
    url = "https://metadata.un.org/thesaurus/alphabetical?lang=en"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    terms = {
        a.text.strip(): a.get('href').split("?")[0]
        for a in soup.select('a.bc-link') 
        if a.get('href') and '/thesaurus/' in a.get('href')
    }

    return terms

def save_unbis_vocab(filepath):
    terms = get_unbis_vocab()
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(terms, f, ensure_ascii=False, indent=4)

def main():
    filepath = 'cache/unbis_vocab.json'
    save_unbis_vocab(filepath)
    print(f"UNBIS vocabulary saved to {filepath}")

if __name__ == "__main__":
    main()