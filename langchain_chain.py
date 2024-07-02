from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SimpleSequentialChain, LLMChain
from langchain.globals import set_debug
import os
from dotenv import load_dotenv

load_dotenv()
set_debug(True)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

interesse = "praia"

modelo_cidade = ChatPromptTemplate.from_template(
    "Dado meu interesse por {interesse}, sugira uma cidade para visitar nas férias. Retorne apenas o nome da cidade:"
)

modelo_restaurantes = ChatPromptTemplate.from_template(
    "Sugira restaurantes populares entre locais em {cidade}"
)

modelo_cultural = ChatPromptTemplate.from_template(
    "Sugira atividades e locais culturais em {cidade}"
)

chain_cidade = LLMChain(prompt=modelo_cidade, llm=llm)
chain_restaurantes = LLMChain(prompt=modelo_restaurantes, llm=llm)
chain_cultural = LLMChain(prompt=modelo_cultural, llm=llm)

chain_main = SimpleSequentialChain(chains=[chain_cidade, chain_restaurantes, chain_cultural], verbose=True)

resultado = chain_main.invoke(interesse)

print(resultado)