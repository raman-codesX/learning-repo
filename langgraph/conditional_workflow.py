from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# STATE

class ReviewState(TypedDict):
    review: str
    neg_break: dict
    sentiment: Literal["positive", "negative"]
    response: str

# STRUCTURED OUTPUT SCHEMAS

class sentiment_format(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the review"
    )


class breakNegative_format(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(
        description="The category of issue mentioned in the review"
    )

    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(
        description="The emotional tone expressed by the user"
    )

    urgency: Literal["low", "medium", "high"] = Field(
        description="How urgent or critical the issue appears to be"
    )


# MODEL

model = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

structured_model1 = model.with_structured_output(sentiment_format)
structured_model2 = model.with_structured_output(breakNegative_format)


# NODES

def find_sentiment(state: ReviewState):

    prompt = f"""
    Analyze the sentiment of this customer review.

    Review: {state["review"]}

    Classify it as positive or negative.
    """

    sentiment = structured_model1.invoke(prompt)

    return {
        "sentiment": sentiment.sentiment
    }


def break_negative(state: ReviewState):

    prompt = f"""
    Diagnose this negative customer review.

    Review:
    {state["review"]}

    Identify:
    1. issue_type
    2. tone
    3. urgency
    """

    neg_break = structured_model2.invoke(prompt)

    return {
        "neg_break": neg_break.model_dump()
    }


def positive_response(state: ReviewState):

    prompt = f"""
    Write a warm thank-you message in response
    to this customer review:

    "{state['review']}"

    Also, kindly ask the user to leave feedback
    on our website.
    """

    response = model.invoke(prompt)

    return {
        "response": response.content
    }


def negative_response(state: ReviewState):

    neg_break = state["neg_break"]

    prompt = f"""
    You are a professional customer support assistant.

    The user had a '{neg_break["issue_type"]}' issue.

    Their emotional tone was '{neg_break["tone"]}'.

    The urgency level is '{neg_break["urgency"]}'.

    Original review:
    {state["review"]}

    Write an empathetic and helpful resolution message.

    Do not claim that the issue has already been fixed.
    """

    response = model.invoke(prompt)

    return {
        "response": response.content
    }


# CONDITIONAL ROUTING

def check_sentiment(
    state: ReviewState
) -> Literal["positive_response", "break_negative"]:

    if state["sentiment"] == "positive":
        return "positive_response"

    return "break_negative"


# GRAPH BUILDER

builder = StateGraph(ReviewState)


# NODES

builder.add_node("find_sentiment", find_sentiment)
builder.add_node("break_negative", break_negative)
builder.add_node("positive_response", positive_response)
builder.add_node("negative_response", negative_response)


# EDGES

builder.add_edge(START, "find_sentiment")

builder.add_conditional_edges(
    "find_sentiment",
    check_sentiment
)

builder.add_edge("break_negative", "negative_response")

builder.add_edge("positive_response", END)
builder.add_edge("negative_response", END)


# COMPILE

workflow = builder.compile()


# INITIAL STATE

initial_state = {
    "review": """
    I've been trying to log in for over an hour now,
    and the app keeps freezing on the authentication screen.

    I even tried reinstalling it, but no luck.

    This kind of bug is unacceptable,
    especially when it affects basic functionality.
    """
}


# EXECUTION

result = workflow.invoke(initial_state)

print("\nSENTIMENT:")
print(result["sentiment"])

print("\nNEGATIVE ISSUE ANALYSIS:")
print(result.get("neg_break", "Not applicable"))

print("\nGENERATED RESPONSE:")
print(result["response"])
