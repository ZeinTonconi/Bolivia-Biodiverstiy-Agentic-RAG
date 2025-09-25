from dotenv import load_dotenv
from crewai import Agent, Crew, Task
from router import decide_travel_guide_coverage
from tools import web_search_tool
from prompts import router_prompt_str, web_prompt_str
from tools import travel_guide_rag_tool

load_dotenv()

user_query_str = input("Ask me anything about traveling in Bolivia (or general): ")

router_agent = Agent(
    role="Router",
    goal="Decide the best tool (Travel Guide, Web Search, or General LLM) for the query",
    backstory="""You are an intelligent router that analyzes the user query.
If it's about Bolivia travel, use the Travel Guide tool.
If it's about recent events or outside the guidebook, use the Web Search. 
You do not give explanation about your answer, you only give the correct tool""",
    tools=[], 
    prompt_template=router_prompt_str,
    # llm="gpt-4o-mini"
)

router_task = Task(
    description=f"User query: {user_query_str}",
    expected_output="Short reasoning note and explicit: Chosen Tool: <tool name>",
    agent=router_agent,
)

if decide_travel_guide_coverage(user_query_str):
    chosen_tool = "Bolivia Travel guide"
else:
    chosen_tool = "Web Search"

print(chosen_tool)

if chosen_tool.lower().startswith("bolivia"):

    travel_expert = Agent(
        role="Expert travel guide",
        goal="Provide useful information and recommendations about different travel destination.",
        backstory="""You are a experienced travel specialized in Bolivia. 
        Your knowledge allows you to offer very useful advice to travelers that want to visit Bolivia.""",
        tools=[travel_guide_rag_tool]
    )

    get_info_task = Task(
        description=f"Get precise and relevant information about the user query using the provided travel guide tool. User query: {user_query_str}",
        expected_output="A brief summary with details, facts and tips about the places to travel.",
        agent=travel_expert,
    )

    recommend_task = Task(
        description=f"""Recommend a list of places of interest to visit or activities
        to do in a given city in Bolivia using the original user query: {user_query_str}, 
        and the information obtained from the previous results""",
        expected_output="""A detailed report of places or activities in a bullet list.
        Include information such as availability, price range, popularity""",
        agent=travel_expert,
    )

    crew = Crew(
        agents=[travel_expert],
        tasks=[get_info_task, recommend_task],
        verbose=True
    )

    result = crew.kickoff()

    print(result)

    print("\n=== Final Answer (Travel Guide) ===\n", result)

else:
    snippet_1, url_1, snippet_2, url_2 = web_search_tool.run(user_query_str)
    web_agent = Agent(
        role="WebSynthesizer",
        goal="Synthesize a short answer using only the two provided web snippets",
        backstory="You are concise and must strictly use only provided snippets.",
        tools=[],  # no tools needed; we pass snippets in the task description
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
        "Note: This information is outside the travel guide and was gathered quickly from web sources.\n\n"
        + model_answer.strip()
        + "\n\nLinks:\n" + url_1 + ("\n" + url_2 if url_2 else "")
    )
    print("\n=== Final Answer (Web Search) ===\n",final)

