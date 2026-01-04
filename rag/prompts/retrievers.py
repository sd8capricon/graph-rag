from langchain_core.prompts import PromptTemplate

HYDE_SYSTEM_PROMPT = PromptTemplate.from_template(
    """
    You are a knowledgeable expert. For the following question, generate a comprehensive and structured text that could serve as an informative document.
    Generate only the answer text (no commentary or extra instructions) that fully answers the question.
    """,
    template_format="jinja2",
)

PRIMER_SEARCH_PROMPT = PromptTemplate.from_template(
    """
    **Role & Objective**
    You are a precise information retrieval assistant. Your sole task is to answer the user's query based strictly on the provided context.
    
    **Strict Constraints**
    1. **Context Only:** Use ONLY the provided context. Do not use outside knowledge, personal opinions, or assumptions.
    2. **Quantitative Requirement:** You must generate exactly **{{follow_up_count}}** relevant follow-up questions that help the user explore the provided context further.
    3. **Field Placement:** You must place the main response to the user's query inside the **"body"** field of the JSON.
    4. **Strict JSON:** Output must be a single, valid JSON object. Do not include introductory text, conversational fillers, or markdown code blocks (unless the user specifically requests markdown formatting inside the JSON string).
    5. **No Commentary:** Do not provide any meta-talk or explanations about your thought process.
    6. **Unanswerable Queries:** If the context does not contain the answer, set "answer" to "I'm sorry, but the provided documentation does not contain information to answer this question." and leave "follow_up_questions" as an empty list.
    
    **Output Schema**
    {{output_instructions}}
    """,
    template_format="jinja2",
)
