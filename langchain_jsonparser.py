from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SimpleSequentialChain, LLMChain
from langchain.globals import set_debug
import os
from dotenv import load_dotenv
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
set_debug(True)

class Destino(BaseModel):
    cidade = Field("Cidade a visitar")
    motivo = Field("Motivo pelo qual é interessante visitar a cidade")


llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

parseador = JsonOutputParser(pydantic_object=Destino)

modelo_cidade = PromptTemplate(
    template= """Sugira uma cidade dado meu interesse por {interesse}
    {formatacao_de_saida}
    
    """,
    input_variables=["interesse"],
    partial_variables={"formatacao_de_saida": parseador.get_format_instructions()}
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

chain_main = SimpleSequentialChain(chains=[chain_cidade, chain_restaurantes, chain_cultural
                                           ], 
                                           verbose=True)

resultado = chain_main.invoke("museus")

print(resultado)