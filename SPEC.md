# Specifications



[World Bank API](https://documents.worldbank.org/en/publication/documents-reports/api)

Consider creating a very simple RAG workflow with PDFs as a baseline to compare any advancements

Develop test questions and answers to evaluate performance for single doc

[Semantic Technology](https://en.wikipedia.org/wiki/Semantic_technology)

[Knowledge Representation & Reasoning](https://en.wikipedia.org/wiki/Knowledge_representation_and_reasoning)

## Tech

### High-level frameworks

For a graph-RAG application, both LlamaIndex and LangChain/LangGraph are suitable options. LlamaIndex is more straightforward for simple RAG apps, while LangGraph provides more control. A robust app might use both.





## Project Epics

1. [Create a KG from the report metadata](#Create a KG from the report metadata)
2. [Stream a file from the API and extract information](#Stream a file from the API and extract information)
3. [Pre-process, chunk and store](#Pre-process, chunk and store)
4. [Extract keywords to use in NER](#Extract keywords to use in NER) 
5. Parse the text with NLP for named entities and a summary
6. Resolve named entities to Wikidata or otherwise (acronyms, WB-specific entities?) Named Entity Linking
7. Update the KG with named entities
8. Add supplementary info for named entities from Wikidata
9. Embed the graph for search and retrieval
10. Build a multi-agent system to query the graph



## Create a KG from the report metadata

See notebooks/world-bank-kg

- Finish all relevant attributes
- Convert to script with number of reports as argument

Returns a `.ttl` file with the KG in RDF format



## Stream a file from the API and extract information

- Research which file parsers are best for PDFs today
- Research how LLMs might inform parsing strategy



A number of Python libraries and APIs are available for parsing PDFs. Many of these have integrations with LangChain ([full list](https://python.langchain.com/docs/integrations/document_loaders/)) or LlamaIndex (pyMuPDF and Unstructured). 

However, Mistral and MinerU create more faithful Markdown representations. 

For this project, I selected MinerU because it did the best at identifying tables and images and it clips an image of each, which can be passed to an LLM for summarization or returned to the user when relevant.





## Pre-process, chunk and store

#### Acronyms

[Schwartz & Hearst 2023](https://www.semanticscholar.org/paper/A-Simple-Algorithm-for-Identifying-Abbreviation-in-Schwartz-Hearst/44e9739e35a80c71e61b4e08871585ba75da2d2b) is common algorithm, implemented in `scispacy`, but I add some cleaning. Last time I just copied the code maybe due to dependency issues, might do that again.

Do this first.

#### Co-references

Reference resolution (He > Greg) [paper](https://arxiv.org/pdf/2312.06648.pdf).

Both [SpanBERT](https://arxiv.org/abs/1907.10529) and AllenNLP offer valuable tools for coreference resolution, with SpanBERT being more specialized for longer documents while AllenNLP provides a robust framework for various NLP tasks. 

neuralcoref is depricated. tried coreferee, ok. spacy has an option in experimental.

This is dependency hell, can't do acronyms and coreferences in same library.

It’s often best to not overwrite original text, but to keep a resolved version in parallel (e.g. for KG construction or QA).

#### Chunking

For transformer-based pipelines, we first need to chunk the document to limit the size.

[Mistral AI Basic RAG tutorial](https://docs.mistral.ai/guides/rag/)

[Chunking for RAG: best practices](https://unstructured.io/blog/chunking-for-rag-best-practices)



## Extract keywords to use in NER

### Controlled vocabulary

- [SDG Taxonomy](http://metadata.un.org/sdg)
- [UN Thesaurus](https://research.un.org/en/thesaurus/downloads)
- [IATA Codelists](https://codelists.codeforiati.org/)
- OECD DAC-CRS Purpose Codes
- H[umanitarian Exchange Language](https://hxlstandard.org/) (HXL)
- [Relief Web Taxonomy as a Service](https://reliefweb.int/taxonomy-service)

- [LinkedSDGs API](https://linkedsdg.officialstatistics.org/#/api) (see [LOD4Stats repo](https://github.com/UNStats/LOD4Stats) for implemented SDG Turtle files)

### Triple extraction

Use spacy to extract triples (subject, predicate, object) for document-specific knowledge graph



Now that we have text in Markdown, we can begin the keyword extraction process. Keywords augment named entities from NLP pipelines with domain-specific concepts (e.g., environmental crime for the international development sector).

While a controlled vocabulary could be developed by hand, algorithms like TextRank and [RAKE](https://github.com/aneesha/RAKE) and [YAKE!](https://github.com/LIAAD/yake) are commonly used to extract keywords. With transformers, we have available models like [KeyBERT](https://github.com/MaartenGr/KeyBERT).

However, frontier language models like GPT-4 can likely do even better than these specialized models when prompted. 





Use EntityRuler. You should at least ensure they have a wikidata entry. See ESSC research on Named Entity Linking. Create a controlled vocabulary.

