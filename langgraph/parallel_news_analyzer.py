from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

class NewsState(TypedDict):
    article : str
    facts : str
    bias_analysis : str
    final_report : str

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

def fact_extractor(state: NewsState):
        prompt = f"""
    Analyze the following news article.

    Extract:
    1. Main event
    2. People or organizations involved
    3. Important dates and locations
    4. Key facts

    Only extract information present in the article.
    Do not invent facts.

    Article:
    {state['article']}
    """
        response = llm.invoke(prompt)
        return {"fact" : response.content}

def bias_analyzer(state: NewsState):
      prompt = f"""
    Analyze the language of this news article.

    Identify:
    1. Emotionally charged words
    2. Potentially biased statements
    3. Neutral or factual language
    4. Overall tone

    Do not assume bias without evidence.
    If no clear bias is found, say so.

    Article:
    {state['article']}
    """
      response = llm.invoke(prompt)
      return {"bias_analysis" : response.content}

def combine_result(state: NewsState):
      report = f"""
    ===== NEWS ANALYSIS REPORT =====

    ORIGINAL ARTICLE:
    {state['article']}

    FACT EXTRACTION:
    {state['facts']}

    BIAS ANALYSIS:
    {state['bias_analysis']}

    ===== END OF REPORT =====
    """
      return {"final_report" : report}

builder = StateGraph(NewsState)

builder.add_node("fact_extractor", fact_extractor)
builder.add_node("bias_analyzer", bias_analyzer)
builder.add_node("combine_result", combine_result)

builder.add_edge(START, "fact_extractor")
builder.add_edge(START, "bias_analyzer")

builder.add_edge("fact_extractor", "combine_result")
builder.add_edge("bias_analyzer", "combine_result")

builder.add_edge("combine_result", END)

graph = builder.compile()

article = """
The government announced a new public transport policy on Monday.
The policy aims to improve city transportation.
However, several citizens criticized the decision,
calling it a disappointing move that ignores their concerns.
"""

result = graph.invoke({
    "article": article,
    "facts": "",
    "bias_analysis": "",
    "final_report": ""
})

print(result["final_report"])
