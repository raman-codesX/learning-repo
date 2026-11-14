from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class BMI(TypedDict):
    hight : float
    weight : float
    bmi : float

def bmi_calculate(state: BMI):
    hight = state["hight"]
    weight = state["weight"]

    bmi = weight/(hight**2)
    return {"bmi": round(bmi, 2)}

builder = StateGraph(BMI)

builder.add_node("bmi_calculate", bmi_calculate)

builder.add_edge(START, "bmi_calculate")
builder.add_edge("bmi_calculate", END)

graph = builder.compile()

result = graph.invoke({"hight": 1.75, "weight": 54})

print(result)