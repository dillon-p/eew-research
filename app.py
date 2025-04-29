import streamlit as st

from sqlalchemy import create_engine
from pathlib import Path
import sqlite3
import pandas as pd
import glob

from typing_extensions import TypedDict
from typing import Annotated

from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDataBaseTool
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

from st_callback_util import get_streamlit_cb

from dotenv import load_dotenv
import zipfile

load_dotenv()

data_path = "data"
zip_path = "generated.zip"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(data_path)

conn = sqlite3.connect('eew.db')

csvs = glob.glob(f"{data_path}/generated/*.csv")
# print(csvs)
for csv in csvs:
    if csv == 'data/generated/rcra_violations.csv':
        df = pd.read_csv(csv)
        df['REPORTING_YEAR'] = pd.to_datetime(df['DATE_VIOLATION_DETERMINED'], format='%Y-%m-%d').dt.year
    else:
        df = pd.read_csv(csv)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    table_name = csv.split("/")[-1].replace(".csv", "")
    df.to_sql(table_name, conn, if_exists='replace', index=False)
conn.close()

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.2,
    streaming=True
)

db = SQLDatabase.from_uri("sqlite:///eew.db")

sql_prompt_text = '''
You are a SQLite expert. Given an input question, follow these steps to create a syntactically correct {dialect} query.

1. **Identify the tables that contain the relevant data:**
    - DO NOT make up tables, use the ones below.
    - The table `county_political_reporting` contains information about the party in control of counties.
    - The table `rcra_violations` contains information about federally regulated facilities who have RCRA violations. Columns like `FAC_STATE`, `FAC_COUNTY`, and `FACILITY_NAME` are useful for filtering while `VIOLATION_TYPE_DESC` can be used to count the number of violations.
    - The table `toxic_emissions` contains information about toxic pollutant emissions from the facilities. Columns like `FAC_STATE`, `FAC_COUNTY`, `FACILITY_NAME` and `POLLUTANT_NAME` are useful for filtering while `ANNUAL_EMISSIONS` can be used to sum the total amount of emissions.
    - DO NOT use the `toxic_emissions` table if the question is about RCRA violations and DO NOT use the `rcra_violations` table if the question is about toxic pollutant emissions.
2. **Select the Appropriate Columns:**
   - Use only the columns relevant to the question.
   - `year` or `REPORTING_YEAR` can be used for dates.
3. **Understand any Filtering:**
   - Pay close attention to any filtering criteria in the question. For example, if the question specifies a particular year, state, county, facility name or pollutant ensure that the SQL query includes a `WHERE` clause to filter the data accordingly.
4. **Construct the SQL Query:**
   - Use the simplest query as possible to answer the question.
   - Use aggregate functions like `AVG`, `MAX`, or `MIN` as needed using `GROUP BY`.
   - If a column has `count` in the name, then it should be summed.
   - If the question involves ranking (e.g., "top 5 violators"), use `ORDER BY` with the appropriate column.
5. **Limit the Results:**
   - Use the `LIMIT` clause to restrict the number of results to {top_k} if the total number of results exceeds 50, unless specified otherwise.

Return ONLY the SQLQuery with no other context given. Do not include the question in the output. Do not include the title 'SQLQuery' in the output.

    Use the following format:
    Question: Question here
    SQLQuery: SQL Query to run
    SQLResult: Result of the SQLQuery
    Answer: Final answer here
    
    Only use the following tables:
    {table_info}

Below are a number of examples of questions and their corresponding SQL queries. 
Use these examples to help you understand how to construct your own SQL queries. But do not copy them directly. 
Instead, use them as a guide to create your own SQL queries based on the question you are given.
'''
examples = [
        {
                'input': "Which party was in control in Maricopa county in 2020?",
                'query': "SELECT party_in_control FROM county_political_reporting WHERE county = 'MARICOPA' AND year = 2020;",
        },
        {
                'input': "What are the total number of RCRA violations by state?",
                'query': "SELECT FAC_STATE, count(*) as avg_violations FROM rcra_violations GROUP BY state;",
        },
        {
                'input': "In 2018, how much much Ammonia was released in Illinois?",
                'query': "SELECT sum(ANNUAL_EMMISIONS) FROM toxic_emissions WHERE POLLUTANT_NAME = 'Ammonia' AND FAC_STATE = 'IL' AND REPORTING_YEAR = 2018;",
        },
        {
                'input': "Give me the names of the top 5 facilities in Texas by the number of RCRA violations",
                'query': "SELECT FAC_NAME, COUNT(*) as total_violations FROM rcra_violations WHERE FAC_STATE = 'TX' GROUP BY FAC_NAME ORDER BY total_violations DESC LIMIT 5;",
        }
]

# sql_chain_prompt = PromptTemplate.from_template((sql_prompt_text + '\nQuestion: {input}'))

example_prompt = PromptTemplate.from_template("Example User input: {input}\nExample SQLQuery: {query}")
# Below are a number of examples of questions and their corresponding SQL queries. 
# Use these examples to help you understand how to construct your own SQL queries. But do not copy them directly. 
# Instead, use them as a guide to create your own SQL queries based on the question you are given.
sql_agent_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=sql_prompt_text,
        suffix="Question: {input}\nSQLQuery:",
        input_variables=["input", "table_info", "dialect", "top_k"],
)

print(sql_agent_prompt)

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
    
class QueryOutput(TypedDict):
    """Generate SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = sql_agent_prompt.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

# print(write_query({'question': "How much of the pollutant Ammonia was emitted in 2020?"}))

def execute_query(state: State):
    """Execute SQL query and return results."""
    execute_query_tool = QuerySQLDataBaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

# print(execute_query({'query': "SELECT county, state, sum(rcra_violations_count) as total_violations FROM rcra_violations_emissions GROUP BY county, state ORDER BY total_violations DESC LIMIT 5;"}))

def generate_answer(state: State):
    """Answer question using retrieved information from SQL query as context."""
    prompt = (
        "Given the following user question, corresponding SQL query and SQL result, answer the user question.\n\n"
        f"Question:{state['question']}\n"
        f"SQL Query: {state['query']}\n"
        f"SQL Result: {state['result']}\n\n"
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence(
    [
        write_query,
        execute_query,
        generate_answer,
    ]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()

def invoke_sql_graph(st_messages, callables):
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")
    return graph.invoke({"question": st_messages}, config={"callbacks": callables})

st.title("Toxic Pollutants and RCRA Violations Assistant")
st.image("images/banner_image.png", use_container_width=False)
st.write(
    """
    Hello! \n
    I am a bot that can help you with questions about Toxic Pollutant emissions and Resource Conservation and Recovery Act (RCRA) violations. \n
    You can ask me about the following topics: \n
    - Toxic Pollutant emissions by state, county, or facility name
    - RCRA violations by state, county, or facility name
    """
)

# if "expander_open" not in st.session_state:
#     st.session_state.expander_open = True

# with st.expander(label="Simple bot", expanded=st.session_state.expander_open):
#     """A simple chatbot to 
#     answer your questions about Toxic Pollutant emissions 
#     and Resource Conservation and Recovery Act violations.
#     """

if prompt := st.chat_input("What would you like to know?"):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = get_streamlit_cb(st.container())
        response = invoke_sql_graph(
            prompt,
            callables=[st_callback],
        )
        st.markdown("...")
        # st.write(response)

# if prompt is not None:
#     st.session_state.expander_open = False

# if "messages" not in st.session_state:
#     st.session_state['messages'] = [AIMessage(content='How can I help you?')]

# for msg in st.session_state.messages:
#     if isinstance(msg, AIMessage):
#         st.chat_message("assistant").write(msg.content)
#     elif isinstance(msg, HumanMessage):
#         st.chat_message("user").write(msg.content)

# if prompt:
#     st.session_state.messages.append(HumanMessage(content=prompt))
#     st.chat_message("user").write(prompt)

#     with st.chat_message("assistant"):

#         st_callback = get_streamlit_cb(st.container())
