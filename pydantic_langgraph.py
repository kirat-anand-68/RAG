### Pydantic data validation

from langgraph.graph import StateGraph,START,END
from pydantic import  BaseModel

class State(BaseModel):## pydantic during the runtime check whether we are passing the right vcalue or not
    name:str

## node function
def example_node(state:State):
    return {"name":"Hello"}

## StateGraph
builder=StateGraph(State)
builder.add_node("example_node",example_node)

builder.add_edge(START,"example")
builder.add_edge("example_node",END)

graph=builder.compile()
graph.invoke({"name":"Krish"})
