# Specifications

Domain-specific LM.

[World Bank API](https://documents.worldbank.org/en/publication/documents-reports/api)

In the course of this project the author has explored a variety of available implementations and frameworks for individual pipeline components. A review of options and justification for choices made in this pipeline can be found in Appendix A. Ultimately, the purpose of this project is to explore the ability of a GraphRAG-style solution to surface synthesized findings from a domain-specific corpus.

## Considerations

### High-level frameworks

For a graph-RAG application, both LlamaIndex and LangChain/LangGraph are suitable options. LlamaIndex is more straightforward for simple RAG apps, while LangGraph provides more control. A robust app might use both.

### LLM integrations

A KG can be built entirely by prompting. see this [demo](https://medium.com/@john011334/transforming-unstructured-text-into-interactive-knowledge-graphs-with-large-language-models-82f5060ebd8c). Compare LLM prompt method with traditional NLP methods at each step. "Murder your (pipeline) darlings".

### Linked Data

Can use RDF to create formal links between nodes (with `instanceOf` QID) and relationships (with SKOS). Or not. What is the advantage/disadvantage?

### Cost

Understanding cost at each step in the pipeline will be important before running the model on a large corpus.

## Practices

Create an evaluation baseline first

- Develop test questions and answers to evaluate performance for small cluster of docs (one geography or project or funder)

Create an LLM powered alternative also that can be used for comparison. Consider cost and accuracy.

Visualize whenever possible (embedding space, lexical graph, property graph)

It’s often best to not overwrite original text, but to keep a resolved version in parallel (e.g. for KG construction or QA).

To organize new pipeline components and Q&A solutions, it can be useful to think of the set of problems each pipeline component is trying to solve. Rather than build a modular pipeline with swappable components, select a combination of frameworks that address all relevant problems. 

- For example, in RAG the retrieval unit is typically the "chunk". Problems can arise if the chunk is not contextualized and self-contained (e.g., an acronym without its definition or a coreference without the proper noun). This is typically addressed by pipeline components for abbreviation resolution, coreference resolution, and named entity recognition. A further step is entity resolution and named entity linking to a KB like wikipedia.
- Proposition-based retrieval units solve this at inference time, and should resolve both coreferences and acronyms. (Chen et al 2024). NEL can be done as a downstream task on the propositions. 
- In contrast, we can rely on graph embeddings to cluster the coreferences and the acronyms together with other instances of the same entity (Edge et al 2024) so we don't need to resolve entity references for inference. NEL could be tacked on by resolving one or more  entities in the embedded space, but it 

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

Co-references need not necessarily be resolved if the graph is embedded as similar entities will be embedded near each other (Edge 2024). 

However, some use cases might suffer, especially those that  

[Proposition retriever](https://arxiv.org/pdf/2312.06648.pdf) showed promising improvements over sentence level or passage level retrieval in Q&A tasks by resolving text into discrete propositions (statements of facts), which is a method of co-reference resolution. although the authors did not propose this, the approach is especially suitable for representation in a knowledge graph as each proposition could be modeled as a SPO triple. The *Propositionizer* model is available on HuggingFace [https://huggingface.co/chentong00/propositionizer-wiki-flan-t5-large](https://huggingface.co/chentong00/propositionizer-wiki-flan-t5-large). 

Both [SpanBERT](https://arxiv.org/abs/1907.10529) and AllenNLP offer valuable tools for coreference resolution, with SpanBERT being more specialized for longer documents while AllenNLP provides a robust framework for various NLP tasks. 

neuralcoref is depricated. tried coreferee, ok. spacy has an option in experimental.

This is dependency hell, can't do acronyms and coreferences in same library.



#### Chunking

For transformer-based pipelines, we first need to chunk the document to limit the size.

[Mistral AI Basic RAG tutorial](https://docs.mistral.ai/guides/rag/)

[Chunking for RAG: best practices](https://unstructured.io/blog/chunking-for-rag-best-practices)



## Extract keywords to use in NER

Named entities exhibit long-tail behavior (most entities have very few occurences in the corpus) (Cheng 2024). Proposition retriever improves performance for low-frequency entities.

### Controlled vocabulary

- [SDG Taxonomy](http://metadata.un.org/sdg)
- [UN Thesaurus](https://research.un.org/en/thesaurus/downloads)
- [IATA Codelists](https://codelists.codeforiati.org/)
- OECD DAC-CRS Purpose Codes
- H[umanitarian Exchange Language](https://hxlstandard.org/) (HXL)
- [Relief Web Taxonomy as a Service](https://reliefweb.int/taxonomy-service)

- [LinkedSDGs API](https://linkedsdg.officialstatistics.org/#/api) (see [LOD4Stats repo](https://github.com/UNStats/LOD4Stats) for implemented SDG Turtle files)

### Corpus-specific vocabulary

While a controlled vocabulary could be developed by hand, algorithms like TextRank and [RAKE](https://github.com/aneesha/RAKE) and [YAKE!](https://github.com/LIAAD/yake) are commonly used to extract keywords. With transformers, we have available models like [KeyBERT](https://github.com/MaartenGr/KeyBERT).

However, frontier language models like GPT-4 can likely do even better than these specialized models when prompted. 

### Triple extraction

Use spacy to extract triples (subject, predicate, object) for document-specific knowledge graph



Now that we have text in Markdown, we can begin the keyword extraction process. Keywords augment named entities from NLP pipelines with domain-specific concepts (e.g., environmental crime for the international development sector).



### Predicates

How does using SKOS help or hurt our use case for predicates? Is there another controlled vocabulary that has predicates?



Use EntityRuler. You should at least ensure they have a wikidata entry. See ESSC research on Named Entity Linking. Create a controlled vocabulary.

