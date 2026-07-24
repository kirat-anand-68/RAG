from dotenv import load_dotenv

from typing_extensions import TypedDict
from typing import Annotated

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    AnyMessage
)

from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

import matplotlib.pyplot as plt
from PIL import Image
import io


# =========================================================
# 1. LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# 2. CREATE CONVERSATION HISTORY
# =========================================================

messages = [
    AIMessage(
        content="Please tell me how can I help",
        name="LLMModel"
    ),

    HumanMessage(
        content="I want to learn coding",
        name="Kirat"
    ),

    AIMessage(
        content="Which programming language do you want to learn?",
        name="LLMModel"
    ),

    HumanMessage(
        content="I want to learn Python programming language",
        name="Kirat"
    )
]


# Print conversation
for message in messages:
    message.pretty_print()


# =========================================================
# 3. CREATE LLM
# =========================================================

llm_groq = ChatGroq(
    model="llama-3.3-70b-versatile"
)


# Send complete conversation to LLM
response = llm_groq.invoke(messages)

print(response.content)


# =========================================================
# 4. CREATE TOOL
# =========================================================

def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        Sum of a and b.
    """

    return a + b


# Bind tool with LLM
llm_tools = llm_groq.bind_tools([add])


# Test tool calling
tool_call = llm_tools.invoke(
    [
        HumanMessage(
            content="What is 2 plus 2?",
            name="Kirat"
        )
    ]
)


print(tool_call.tool_calls)


# =========================================================
# 5. DEFINE LANGGRAPH STATE
# =========================================================

class State(TypedDict):

    messages: Annotated[
        list[AnyMessage],
        add_messages
    ]


# =========================================================
# 6. UNDERSTAND add_messages REDUCER
# =========================================================

initial_messages = [

    AIMessage(
        content="Please tell me how can I help",
        name="LLMModel"
    ),

    HumanMessage(
        content="I want to learn coding",
        name="Kirat"
    )
]


ai_message = AIMessage(
    content="Which programming language do you want to learn?",
    name="LLMModel"
)


# Manually demonstrate reducer
updated_messages = add_messages(
    initial_messages,
    ai_message
)


print(updated_messages)


# =========================================================
# 7. CREATE LANGGRAPH NODE
# =========================================================

def llm_tool(state: State):

    response = llm_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# =========================================================
# 8. BUILD LANGGRAPH
# =========================================================

builder = StateGraph(State)


# Add node
builder.add_node(
    "llm_tool",
    llm_tool
)


# Add edges
builder.add_edge(
    START,
    "llm_tool"
)

builder.add_edge(
    "llm_tool",
    END
)


# =========================================================
# 9. COMPILE GRAPH
# =========================================================

graph = builder.compile()


# =========================================================
# 10. DRAW GRAPH USING MATPLOTLIB
# =========================================================

png_data = graph.get_graph().draw_mermaid_png()

img = Image.open(
    io.BytesIO(png_data)
)


plt.figure(figsize=(8, 5))

plt.imshow(img)

plt.axis("off")

plt.show()


## invocation

messages=graph.invoke({"messages":"What is 2 plus 2"})

for message in messages["messages"]:
    message.pretty_print()
tools=[add]
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

builder=StateGraph(State)

## Add nodes
builder.add_node("llm_tool",llm_tool)
builder.add_node("tools",ToolNode(tools))
## Add Edge
builder.add_edge(START,"llm_tool")
builder.add_conditional_edges(
    "llm_tool",
    # if the latest message (result) from assistant is a tool caal -. tools_condition routes to tools
    # if the latest messaged(result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition
)

builder.add_edge("tools",END)
graph_builder=builder.compile()

png_data = graph.get_graph().draw_mermaid_png()

img = Image.open(
    io.BytesIO(png_data)
)

plt.figure(figsize=(10, 6))

plt.imshow(img)

plt.axis("off")

plt.show()


# =========================================================
# 11. INVOKE GRAPH
# =========================================================

messages=graph.invoke({"messages":"What is 2 plus 2"})

for message in messages["messages"]:
    message.pretty_print()

messages=graph.invoke({"messages":"what is machine learning"})
for message in messages["messages"]:
    message.pretty_print()
    
