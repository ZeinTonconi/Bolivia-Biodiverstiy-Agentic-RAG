from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Crew, Task
from router import decide_biodiversity_coverage
from tools import web_search_tool, biodiversity_rag_tool_wrapper
from prompts import router_prompt_str, web_prompt_str

user_query_str = input("Ask me anything about biodiversity in Bolivia: ")

router_agent = Agent(
    role="Router",
    goal="Decide the best tool (Book Guide or Web Search) for the query",
    backstory="""You are an intelligent router that analyzes the user query.
If it's about Bolivia biodiversity, use the Book Guide tool.
If it's about outside the guidebook, use the Web Search. 
You do not give explanation about your answer, you only give the correct tool""",
    tools=[], 
    prompt_template=router_prompt_str
)

router_task = Task(
    description=f"User query: {user_query_str}",
    expected_output="Short reasoning note and explicit: Chosen Tool: <tool name>",
    agent=router_agent,
)

if decide_biodiversity_coverage(user_query_str):
    chosen_tool = "Book guide"
else:
    chosen_tool = "Web Search"

print(chosen_tool)

if chosen_tool.lower().startswith("book"):

    biodiversity_agent = Agent(
        role="Experto en Biodiversidad",
        goal="Proporciona información útil sobre la biodiversidad de Bolivia",
        backstory="""Eres un especialista en estudios de biodiversidad e informes ecológicos.
            Muestra los hallazgos con precisión utilizando únicamente los documentos de biodiversidad seleccionados proporcionados.
            Das prioridad a la claridad, a la citación fáctica y no inventas detalles faltantes.""",
        tools=[biodiversity_rag_tool_wrapper]
    )

    get_info_task = Task(
        description=f"Obtén información precisa y relevante sobre la consulta del usuario utilizando la herramienta de guía de libros proporcionada. Consulta del usuario: {user_query_str}",
        expected_output="Un texto con detalles, hechos e información sobre la biodiversidad.",
        agent=biodiversity_agent,
    )

    crew = Crew(
        agents=[biodiversity_agent],
        tasks=[get_info_task],
        verbose=True
    )

    overview_result = crew.kickoff()
    overview_text = str(overview_result).strip()
    print("=== Overview ===\n", overview_text)

else:
    snippet_1, url_1, snippet_2, url_2 = web_search_tool.run(user_query_str)
    web_agent = Agent(
        role="WebSynthesizer",
        goal="Synthesize a short answer using only the two provided web snippets",
        backstory="You are concise and must strictly use only provided snippets.",
        tools=[], 
        prompt_template=web_prompt_str,
        llm_kwargs={"temperature": 0, "max_tokens": 120}
    )

    web_task = Task(
        description=f"User query: {user_query_str}\n\nSnippet 1:\n{snippet_1}\n\nSnippet 2:\n{snippet_2}",
        expected_output="Short answer restricted to the snippets provided.",
        agent=web_agent
    )

    crew_web = Crew(agents=[web_agent], tasks=[web_task], verbose=True)
    web_result = crew_web.kickoff()

    model_answer = str(web_result)
    final = (
        "Note: This information is outside the biodiversity guide and was gathered quickly from web sources.\n\n"
        + model_answer.strip()
        + "\n\nLinks:\n" + url_1 + ("\n" + url_2 if url_2 else "")
    )
    print("\n=== Final Answer (Web Search) ===\n",final)

