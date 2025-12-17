import json
from pathlib import Path
import requests

from src.mineru_demo import do_parse

LANGUAGES = {
    'English': 'en'
}

class Reader:
    def __init__(self, output_dir='output', **kwargs):
        self.output_dir = output_dir
        self.kwargs = kwargs


    def read_fn(self, url):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        return response.content


    def process_doc(self, url, file_id, lang):
        pdf_file_names = [str(file_id)]
        file_bytes = self.read_fn(url)
        pdf_bytes_list = [file_bytes]
        p_lang_list = [LANGUAGES.get(lang, 'en')]

        do_parse(
            output_dir=self.output_dir,
            pdf_file_names=pdf_file_names,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=p_lang_list,
            **self.kwargs
        )

        return Path(f'output/{file_id}/auto/{file_id}_content_list.json')

    
    def get_markdown(self, file_id):
        md_file_path = Path(f'{self.output_dir}/{file_id}/auto/{file_id}.md')
        md_text = Path(md_file_path).read_text()

        return md_text


    def get_json(self, file_id):
        json_file_path = Path(f'{self.output_dir}/{file_id}/auto/{file_id}_content_list.json')
        with open(json_file_path, "r", encoding="utf-8") as f:
            return json.load(f)


def main():
    r = Reader()
    url = 'http://documents.worldbank.org/curated/en/099022025104517097/pdf/P17921716becb702318c7f1877bdbd88e20.pdf'
    file_id = 34458416
    file_id = r.process_doc(url, file_id, 'English')

if __name__ == '__main__':
    main()



