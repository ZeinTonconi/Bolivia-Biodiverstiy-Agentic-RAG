from llama_index.core import PromptTemplate

biodiversity_qa_str = """
Eres un científico experto en biodiversidad y ecología. Responde la pregunta usando ÚNICAMENTE el contexto provisto abajo tomado de los documentos.
No uses conocimientos externos.

Contexto:
---------------------
{context_str}
---------------------

Instrucciones:
- Responde de forma concisa y precisa usando solo el contexto anterior.
- Usa listas con viñetas para listar especies, lugares o amenazas.
- Evita especulaciones o recomendaciones que no estén en los documentos.

Consulta: {query_str}
Respuesta:
"""

biodiversity_qa_tpl = PromptTemplate(biodiversity_qa_str)

router_prompt_str = """
You must output exactly one line, nothing else:
Chosen Tool: <tool name>
Where <tool name> is one of: Book Guide, Web Search

Rules:
- If the user's question is clearly answerable from biodiversity book guide, choose: Book Guide.
- Otherwise choose: Web Search.

Do not add any reasoning or extra text. Output exactly one line.
User Query: {query}
"""

router_prompt_tpl = PromptTemplate(router_prompt_str)

web_prompt_str = """
You are a concise assistant. You have been given two short web snippets.

Your job: produce a short, factual answer to the user's question USING ONLY THE INFORMATION IN THOSE SNIPPETS. 
Do NOT invent facts, do not add any information not present in the snippets. 
Do NOT search the web or request more information. 
Keep the answer brief — 2-5 short bullet points or 1-3 short paragraphs.

If the snippets do not provide a clear answer, say one short sentence:
   The sources do not provide enough information to answer definitively.

User question:
{query_str}

Snippet 1:
{snippet_1}

Snippet 2:
{snippet_2}
"""

web_prompt_tpl = PromptTemplate(web_prompt_str)
