


#import libs
import json
import os
from datetime import datetime

import yfinance as yf # Importa a biblioteca yfinance para pegar os dados das ações
from crewai import Agent, Task, Crew, Process # Importa as classes Agent, Task, Crew e Process para utilizar o crew ai

from langchain.tools import Tool # Importa a classe Tool
from langchain_openai import ChatOpenAI # Importa o ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults # Importa a classe DuckDuckGoSearchResults PARA PESQUISAS
# from IPython.display import Markdown
import streamlit as st # Fornece um metodo para contruir aplicação WEB raipido e facil


# In[3]:


#Criando yahoo finance tool
def fetch_stock_price(ticket):
    stock = yf.download(ticket, start='2023-08-08', end='2024-08-08', progress=False)   
    return stock

yahoo_finance_tool = Tool(
    name="Yahoo Finance Tools",
    description="Fetches stocks prices for {ticket} f rom the last year about a specific stock from Yahoo Finance API'",
    func = lambda ticket: fetch_stock_price(ticket)
    )


# In[4]:


# response = yahoo_finance_tool.run("AAPL")


# In[5]:


# print(response)


# In[6]:


#Importando openai
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"] #SUBSTITUIDO PELO SECRETS PARA ENCONDER A CHAVE DE API DO STREAMLIT.p
llm = ChatOpenAI(model="gpt-3.5-turbo")


# In[7]:


stockpriceanalyst = Agent(
    role = "Senior stock price analyst", # Qual o role desse cara, (Função do agente dentro do grupo de Ai)
    goal = "find the {ticket} stock price and analyses trends", # "Qual o objetivo desse agente objetivo especifico"
    backstory = "you're highly experienced in analysing the price  of an specific stock and make predictions about its future  price", # Qual seu backstory (O que você é? Contexto geral sobre o role e oo goal)
    verbose=True,
    llm=llm,
    max_iter = 5,
    memory= True,
    tools = [yahoo_finance_tool],
    allow_delegation = False, # Permite delegação de tarefas entre agentes
)


# In[8]:


Getstockprice = Task(
    description = "analyze the stock {ticket} price history and a trend analyses of up, down or sideways.", # Descritção da tarefa
    expected_output = "Specify the current trend stock price - up, down or sideways. eg. stock='APPL, price UP.'", # Descrição detalhada do que essa tesk quando tiver completa como ela se parece
    agent = stockpriceanalyst # Qual o agente que vai fazer essa tarefa
)


# In[9]:


# Importando a tool do DuckDuckGo
search_tool = DuckDuckGoSearchResults(backend="news", num_results=10)


# In[10]:


NewAnalyst = Agent(
    role = "Stock News Analyst", # Qual o role desse cara, (Função do agente dentro do grupo de Ai)
    goal = """Create a short summary of the market news releted to the stock {ticket} company. Specify the current trend - up, down or sideways with the news context. 
    For each request stock asset, specify a numbet between 0 and 100 , where 0 is extreme fear and 100 is extreme greed.""", # "Qual o objetivo desse agente objetivo especifico"
    backstory = """You're highly experienced in analyzing the market trends and news and have tracked assest for more then 10 years. 
    You're also mster level analyst in the tradicional markets and have deep undestanding of human psychology.
    You understand news, theirs tittles and information, but you look at those with a health dose of skepticism.
    You consider also the source of the news articles.""", # Qual seu backstory (O que você é? Contexto geral sobre o role e oo goal)
    verbose = True,
    llm = llm,
    max_iter = 10, # Número máximo de iterações
    memory = True, # Permite que o agente tenha memória
    tools = [search_tool],
    allow_delegation = False, # Permite delegação de tarefas entre agentes
    )


# In[11]:


getnews = Task(
    description = f"""Take the stock and always include BTC to it (if not request). Use the search tool to search each one individually. 
    The current date is {datetime.now()}.
    Compose the results into a helpfull report.""", # Descritção da tarefa
    expected_output = """A summary of the overall market and one setence summary for each request asset. 
    Include a fear/greed score for each asset based on the news. Use format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE""", # Descrição detalhada do que essa tesk quando tiver completa como ela se parece
    agent = NewAnalyst # Qual o agente que vai fazer essa tarefa
)


# In[12]:


stockanalystwrite = Agent(
    role = "Senior Stock analysts writer", # Qual o role desse cara, (Função do agente dentro do grupo de Ai)
    goal = """Analyze the thends price and write an insighfull compelling and informative 3 paragraph long newsletter based on the stock report and price  trend.""", # "Qual o objetivo desse agente objetivo especifico"
    backstory = """you're widely accepted as the best stock analyst in the market.
    You understand complex concepts and create compelling stories and naratives that resonate with wider audiences.
    You undestand macro factors and combine multiple theories - eg. cycle theory and fundamental analyses. 
    You're able to hold multpli opinions then analyzing anything.""", # Qual seu backstory (O que você é? Contexto geral sobre o role e oo goal)
    verbose=True,
    llm=llm,
    max_iter = 5, # Quantas iterações o agente pode fazer
    memory= True,   # Se o agente tem memória ou não
    allow_delegation = True # Permite delegação de tarefas entre agentes
    )


# In[13]:


writeanalyses = Task(
    description = """Use the stock price trend and the stock news report to create an anlyses and write the newsletter about the {ticket} company thet is brief and highlights the most important points.
    Focus on the stock price trend, news and fear/greed score. what are the near future considerations?
    Include the previous analyses of stock trend and news summry.""", # Descritção da tarefa
    expected_output = """An eloquent 3 paragraphs formated as markdown in an easy readable manner. It should contain: 
    - 3 bullets executive summary 
    - Introduction - set the overall picture and spike up the interest
    - Main part provides the meat of the analysis including the news summary and fead/greed scores
    - summary - key facts and concrete future trend prediction - up, down or sideways.""", # Descrição detalhada do que essa tesk quando tiver completa como ela se parece
    agent = stockanalystwrite, # Qual o agente que vai fazer essa tarefa
    context = [Getstockprice, getnews] # Pega (faz um get) nos contextos das tasks dos outros agentes (Pega os seus outputs das tasks)
)


# In[14]:


#Cria o grupo de agentes
crew = Crew(
    agents= [stockpriceanalyst, NewAnalyst, stockanalystwrite], # Passa os agentes que vão fazer as tasks
    tasks = [Getstockprice, getnews, writeanalyses], # Passa as tasks que os agentes vão fazer
    verbose = True, # Nível de verbose
    process = Process.hierarchical, # Passa a hierarquia que os agentes vão fazer as tasks
    full_output = True, # Se vai mostrar o output completo
    share_crew = False, # Se vai compartilhar o grupo de agentes
    manager_llm = llm, # Passa o llm do manager
    max_iter = 15, # Número máximo de iterações
)


# results = crew.kickoff(inputs=({"ticket": "AAPL"}))

# list(results.keys())


# results(["final_output"])

with st.sidebar:
    st.header("Enter the stock to research")
    with st.form(key="research_form"):
        topic = st.text_input("select the ticket")
        subimit_button = st.form_submit_button(label = "Run Research")
if subimit_button:
    if not topic:
        st.error("Please enter a stock ticket")
    else:
        results = crew.kickoff(inputs={"ticket": topic})
        # st.markdown(results["final_output"])
        # st.markdown(results["final_output"])
        st.subheader("Results of your research")
        st.write(results["final_output"])
# len(results["final_output"])



# Markdown(results["final_output"])

