from sqlalchemy import create_engine
from pathlib import Path
import sqlite3
import pandas as pd
import glob
import zipfile

from typing_extensions import TypedDict
from typing import Annotated

from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

data_path = "data/"
zip_path = "generated.zip"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(data_path)

conn = sqlite3.connect('eew.db')

csvs = glob.glob(f"{data_path}/generated/*.csv")

for csv in csvs:
    df = pd.read_csv(csv)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    table_name = csv.split("/")[-1].replace(".csv", "")
    df.to_sql(table_name, conn, if_exists='replace', index=False)
conn.close()

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.2,
    seed=123,
    streaming=True
)
db = SQLDatabase.from_uri("sqlite:///eew.db")

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

sql_prompt_text = '''
You are a SQLite expert. Given an input question, follow these steps to create a syntactically correct {dialect} query.

1. **Identify the tables that contain the relevant data::**
    - The table `county_political_reporting` contains information about the party in control of counties.
    - The table `rcra_violations` contains information about federally regulated facilities who have RCRA violations. Columns like `FAC_STATE`, `FAC_COUNTY`, and `FACILITY_NAME` are useful for filtering while `VIOLATION_TYPE_DESC` can be used to count the number of violations.
    - The table `toxic_emissions` contains information about toxic pollutant emissions from the facilities. Columns like `FAC_STATE`, `FAC_COUNTY`, `FACILITY_NAME` and `POLLUTANT_NAME` are useful for filtering while `ANNUAL_EMISSIONS` can be used to sum the total amount of emissions.
    - DO NOT use the `toxic_emissions` table if the question is about RCRA violations and DO NOT use the `rcra_violations` table if the quesiton is about toxic pollutant emissions.
2. **Select the Appropriate Columns:**
   - Use only the columns relevant to the question.
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
'''
system_message = PromptTemplate.from_template(sql_prompt_text).format(
    dialect=db.dialect,
    table_info=db.get_table_info(),
    top_k=5,
)

agent_executor = create_react_agent(llm, tools, prompt=system_message)

question = "How many RCRA violations were there in 2020 in the state of California?"
for step in agent_executor.stream(
    {'messages': [{'role':'user', 'content':question}]},
    stream_mode='values'
):
    step['messages'][-1].pretty_print()
