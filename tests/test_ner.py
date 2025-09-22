from pathlib import Path
import pytest
import os
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from ner import EntityExtractor

"""
Requirements
- extract named entities of desired type
"""

text = """
Science shows that honey, rich in antioxidants, can lower a person's cholesterol and promote healing. As a commodity, it can also transform entire communities. Just ask Zoraida Silgado Escobar, legal representative for the Carbebias Honey Producer Association (Asocabebias), in Northern Colombia.

Before beekeeping helped to change her community, her father and neighbors earned their livelihood from the grueling work of unregulated gold mining. Detrimental to health, local governance, and ecosystems, the lower echelons of artisanal and small gold mining (ASGM) can demand much of rural people and offer little in the way of financial return. Laborers at the bottom of the supply chain spend hours bent over, panning in streams and rivers, sifting through mud and pebbles, in precarious working conditions. Then there are the dangers out of the water — armed criminals who often use violence to protect their own mining investments or extort other mining operations.

"Beekeeping gave us an economic alternative to mining," Zoraida said. "My father worked for many years in informal mining, which means my family depended on this activity, like most families in this region." Now Zoraida's father produces honey, just like his daughter.

The training Zoraida and her father received was a component of the USAID Colombia Artisanal Gold Mining-Environmental Impact Reduction Activity — or Oro Legal. This five-year, $20 million project is financed by USAID and implemented by Chemonics. As part of its work, Oro Legal provides alternatives to unregulated gold mining; a sub-sector that has changed Colombia economically and environmentally.
"""

def test_ner_extracts_person_and_org():
    extractor = EntityExtractor()
    ents = extractor.extract(text)

    assert (('Zoraida Silgado Escobar', 'PERSON', (170, 193))) in ents
    assert (('ASGM', 'ORG', (561, 565))) in ents
    assert any(ent for ent in ents if ent[1] == "GPE" and "Colombia" in ent[0])