import urllib.parse, urllib.request, json
import dotenv
from pathlib import Path

env_path = Path("secrets") / ".env"
dotenv.load_dotenv(env_path)
WIKIFIER_KEY = dotenv.get_key(env_path, "WIKIFIER_KEY")
TIMEOUT = 60

def CallWikifier(text, lang="en", threshold=0.8):
    # Prepare the URL.
    data = urllib.parse.urlencode([
        ("text", text), 
        ("lang", lang),
        ("userKey", WIKIFIER_KEY),
        ("pageRankSqThreshold", "%g" % threshold), 
        ("applyPageRankSqThreshold", "true"),
        ("nTopDfValuesToIgnore", "200"), 
        ("nWordsToIgnoreFromList", "200"),
        ("wikiDataClasses", "true"), 
        ("wikiDataClassIds", "false"),
        ("support", "true"), 
        ("ranges", "false"), 
        ("minLinkFrequency", "2"),
        ("includeCosines", "false"), 
        ("maxMentionEntropy", "3")
    ])
    url = "http://www.wikifier.org/annotate-article"
    
    # Call the Wikifier and read the response.
    req = urllib.request.Request(url, data=data.encode("utf8"), method="POST")
    with urllib.request.urlopen(req, timeout = TIMEOUT) as f:
        response = f.read()
        response = json.loads(response.decode("utf8"))

    # Output the annotations.
    if response:
        for annotation in response["annotations"]:
            print("%s (%s)" % (annotation["title"], annotation["url"]))
            print()
            
CallWikifier("""
This document summarizes the environmental impact assessment (EIA) report for a proposed power plant project in Egypt. The report assesses the potential environmental impacts of the project, including noise, flora and fauna, land use, soils, geology, hydrology, traffic, archaeology, historic and cultural heritage, and aquatic environment.\n\n**Main Topics**\n\n1. Environmental Impacts\n2. Noise Assessment\n3. Flora and Fauna Assessment\n4. Land Use, Landscape, and Visual Impacts\n5. Soils, Geology, and Hydrology\n6. Traffic Assessment\n7. Archaeology, Historic, and Cultural Heritage\n8. Aquatic Environment\n\n**Entities**\n\n1. Egyptian Government\n2. World Bank\n3. Power Plant Operator (contractor)\n4. Local communities and stakeholders\n\n**Development Themes**\n\n1. Environmental Sustainability\n2. Social Responsibility\n3. Economic Development\n4. Infrastructure Development\n5. Climate Change Mitigation\n\n**Key Findings**\n\n1. The project is expected to have minimal environmental impacts, with most effects being localized and temporary.\n2. Noise levels during construction and operation will be within Egyptian and World Bank guidelines.\n3. Flora and fauna assessments indicate that the project area has relatively impoverished aquatic habitats, but good site management practices will minimize impacts.\n4. Land use, landscape, and visual impacts are expected to be minor and not significant.\n5. Soils, geology, and hydrology assessments indicate no significant impacts due to the characteristics of the site and proposed mitigation measures.\n6. Traffic assessment indicates that traffic impacts during construction and operation will be insignificant.\n7. Archaeology, historic, and cultural heritage assessments indicate no significant impacts due to the absence of protected areas in the project area.\n\n**Recommendations**\n\n1. Implement good site management practices to minimize environmental impacts.\n2. Monitor water quality and ambient air quality to ensure compliance with Egyptian and World Bank guidelines.\n3. Provide appropriate services for contractors during construction, including relevant water/toilet facilities.\n4. Ensure that the power plant operator meets all regulatory requirements and standards.\n\n**Knowledge Graph Metadata**\n\nTitle: Environmental Impact Assessment Report - Proposed Power Plant Project in Egypt\n\nDescription: This report summarizes the environmental impact assessment (EIA) for a proposed power plant project in Egypt, covering various aspects of environmental sustainability, social responsibility, economic development, infrastructure development, and climate change mitigation.\n\nKeywords: Environmental Impact Assessment, Power Plant Project, Egypt, Sustainability, Social Responsibility, Economic Development, Infrastructure Development, Climate Change Mitigation\n\nEntities:\n\n* Egyptian Government\n* World Bank\n* Power Plant Operator (contractor)\n* Local communities and stakeholders\n\nThemes:\n\n* Environmental Sustainability\n* Social Responsibility\n* Economic Development\n* Infrastructure Development\n* Climate Change Mitigation",
             """)