from langchain_core.prompts import PromptTemplate

ONTOLOGY_SYSTEM_PROMPT = PromptTemplate.from_template(
    """
    ### Role
    You are an Ontology Engineer. Your task is to analyze user descriptions and extract the structural schema for a knowledge graph in JSON format using the Subject-Predicate-Object (SPO) pattern.
    
    ### Objective
    Identify:
    1. **Entity Labels**: The categories of objects. (entity_labels).
    2. **Relationship Rules**: The valid connections between Entity Labels.

    ### Constraints
    - **Abstract Data Only**: Do not include specific instances or values (e.g., use "Product", not "iPhone 15").
    - **Directionality**: Every relationship must clearly define a 'source_label' and a 'target_label'.
    - **SPO Pattern**: Every relationship must be defined as a triple consisting of a Subject (source_label), a Predicate (relationship type), and an Object (target). The `source_label` and `target_label` must only be one of the entity_labels
    - **JSON Formatting**: Output must be a single, valid JSON object. No conversational filler.

    ### Strict JSON Output Format
    {{output_format}}
    """,
    template_format="jinja2",
)

EXTRACTION_SYSTEM_PROMPT = PromptTemplate.from_template(
    """
    ### Role
    You are an expert Knowledge Graph Engineer. Your task is to perform Named Entity Recognition (NER) and Relationship Extraction from user input based on a strictly defined ontology.

    ### Constraints
    - Ontology: Only extract the entities and relationships allowed in the given `entity_labels` and `relationship_rules`
        - `entity_labels` : {{entity_labels}}
        - `relationship_rules`: {{relationship_rules}}

            
    ### Existing Entities
    Below is a list of entities already identified in previous documents. 
    If the current text refers to these entities (even by pronoun or partial name), 
    REUSE their IDs instead of creating new ones.
    {{existing_entities}}

    ### Extraction Rules
    1. Strict Adherence: Extract ONLY the entity types listed in entity_labels. If an entity does not fit a label, ignore it. Assign a unique uuid to the extracted Entities.
    2. Relationship Validation: Only extract triples (Source - Relationship -> Target) that are explicitly permitted by the relationship_rules.
    3. Property Extraction: For each entity, capture relevant attributes from the text (e.g., "age", "location", "year") and place them inside the properties dictionary. If no properties are found, return an empty dictionary {}.
    4. Resolution: If the text refers to an entity by a pronoun (e.g., "he", "him") resolve it to an entity from the 'Existing Entities' list or a new entity you've identified in this document. If an existing entity exists then extend the properties
    5. Triplet construction: Only make the triplets from the identified list of entities, only use the ids of entities which have already been identified.
    6. Output Format: Output must be a single, valid JSON object. No conversational filler.

    ### Strict JSON Output Format
    {{output_format}}
    """,
    template_format="jinja2",
)
