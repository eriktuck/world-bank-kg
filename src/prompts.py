entity_linker_prompt = """
For the list of entities provided below, match each entity with the closest QID from WikiData.

Each entity is presented as a comma separated list in the format `entity_name (entity_type)` where entity_name is the name of the entity as found in the text and entity_type can be one of
- PERSON: People, including fictional characters.
- NORP: Nationalities, religious groups, political groups.
- FAC: Facilities: buildings, airports, highways, bridges, etc.
- ORG: Organizations: companies, agencies, institutions.
- GPE: Countries, cities, states (geo-political entities).
- LOC: Locations that are not GPEs, e.g. mountain ranges, bodies of water.
- PRODUCT: Objects, vehicles, foods, devices.
- EVENT: Named events: wars, sports events, hurricanes.
- WORK_OF_ART: Titles of books, songs, artworks.
- LAW: Named documents made into laws.
- LANGUAGE: Named languages.
- ACRONYM: An acronym.
- CUSTOM: A custom entity declared by the document itself.

Use the entity_type to help disambiguate multiple QIDs.
Some entities may be misspelled, please search WikiData based on the correct spelling. 
Some entities may be close synonyms of WikiData entities, please return the 

Also return a confidence score for each entity on a scale from 1 to 5, 1 being very low confidence and 5 being an exact match.

If you cannot make a match with any level of confidence, simply return 'None' and 'None' for QID and confidence, respectively.

Return ONLY valid Python code:
A list of 4-tuples, where each tuple is structured as:
(entity_name: str, entity_type: str, QID: str, confidence: int)

Return a JSON object that can be decoded to the above format with json.loads(response.choices[0].message.content).

Do not include any explanation, markdown fences, or extra text.

Before returning the list, make sure that every entity in the list provided is also in the returned list. If an entity is missing, add it back with 'None' and 'None' for QID and confidence, respectively
"""