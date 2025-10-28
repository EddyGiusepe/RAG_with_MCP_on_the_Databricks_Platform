#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Script server.py
================
Este script cria um servidor MCP que carrega documentos de uma pasta,
indexa esses documentos usando um índice vetorial (``VectorStoreIndex``)
e expõe uma API para receber consultas e gerar respostas baseadas no
modelo OpenAI (``GPT-4.1-nano``) usando o ``LlamaIndex``.

Run
---
uv run server.py
"""
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field
from litserve.mcp import MCP
import litserve as ls
import os
from typing import Dict, Any
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv())  # read local .env file
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


model = "gpt-4.1-nano"  # "gpt-4o"


class RequestType(BaseModel):
    """
    Modelo de requisição para consultas ao sistema RAG.

    Esta classe define a estrutura de dados esperada nas requisições HTTP
    enviadas ao servidor MCP. Utiliza Pydantic para validação automática
    dos dados de entrada.

    Attributes
    ----------
    query : str
        A consulta ou pergunta em linguagem natural que será processada
        pelo sistema RAG (Retrieval-Augmented Generation). Este campo é
        obrigatório e deve conter uma string não vazia que representa a
        pergunta do usuário sobre o currículo do Dr. Eddy Giusepe Chirinos Isidro.

    Examples
    --------
    >>> request = RequestType(query="Qual é a formação acadêmica do Dr. Eddy?")
    >>> print(request.query)
    'Qual é a formação acadêmica do Dr. Eddy?'

    Notes
    -----
    - A validação automática garante que o campo `query` seja sempre uma string
    - O Pydantic levantará uma exceção ValidationError se o tipo estiver incorreto
    - Esta classe é usada pelo método `decode_request` da API DocumentChatAPI
    """

    query: str = Field(
        ...,
        description="Consulta em linguagem natural para busca no sistema RAG",
        min_length=1,
        examples=[
            "Qual é a experiência profissional do Dr. Eddy?",
            "Quais são as habilidades técnicas mencionadas?",
            "Onde o Dr. Eddy trabalha atualmente?",
        ],
    )


class DocumentChatAPI(ls.LitAPI):
    """
    API de Chat com documentos usando RAG (Retrieval-Augmented Generation).

    Esta classe implementa uma API LitServe que processa consultas sobre
    documentos indexados, usando LlamaIndex e OpenAI para gerar respostas
    contextualizadas.
    """

    def setup(self, device: str) -> None:
        """
        Inicializa o modelo LLM e cria o índice vetorial dos documentos.

        Este método é chamado automaticamente pelo LitServe quando o servidor
        inicia. Carrega o modelo OpenAI, indexa os documentos e prepara o
        query engine.

        Parameters
        ----------
        device : str
            Identificador do dispositivo (CPU/GPU) fornecido pelo LitServe.
        """
        Settings.llm = OpenAI(
            api_key=OPENAI_API_KEY,
            temperature=0.0,
            model=model,
            reasoning_effort="medium",
            system_prompt="""Você é um assistente educado e gentil chamado 'BotPerfil' que responde perguntas 
                             sobre o currículo/documento do Dr. Eddy Giusepe Chirinos Isidro. Só deve responder
                             dentro do contexto do currículo/documento e na língua portuguesa do Brasil (pt-br).
                             Se a pergunta não estiver dentro do contexto do currículo/documento, você deverá
                             dizer que só responde perguntas relacionadas ao currículo/documento do Dr. Eddy 
                             Giusepe Chirinos Isidro. NUNCA deve responder perguntas que não estejam dentro do
                             contexto do currículo/documento.
                          """
        )
        documents = SimpleDirectoryReader(
            "/home/eddygiusepe/2_GitHub/RAG_with_MCP_on_the_Databricks_Platform/data"
        ).load_data(show_progress=True)
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        self.query_engine = index.as_query_engine()

    def decode_request(self, request: RequestType) -> str:
        """
        Extrai a query da requisição HTTP.

        Parameters
        ----------
        request : RequestType
            Objeto Pydantic contendo os dados validados da requisição.

        Returns
        -------
        str
            A string de consulta extraída da requisição.
        """
        return request.query

    def predict(self, query: str) -> Any:
        """
        Processa a consulta e gera resposta usando o sistema RAG.

        Busca documentos relevantes e usa o modelo LLM para gerar uma
        resposta contextualizada.

        Parameters
        ----------
        query : str
            A consulta em linguagem natural a ser processada.

        Returns
        -------
        Response
            Objeto Response do LlamaIndex com a resposta gerada.
        """
        return self.query_engine.query(query)

    def encode_response(self, output: Any) -> Dict[str, Any]:
        """
        Formata a resposta do modelo em JSON para o cliente.

        Parameters
        ----------
        output : Response
            Objeto Response do LlamaIndex contendo a resposta.

        Returns
        -------
        Dict[str, Any]
            Dicionário com a resposta formatada para JSON.
        """
        return {"output": output.response}


if __name__ == "__main__":
    api = DocumentChatAPI(
        mcp=MCP(
            description="Respostas a perguntas sobre o curriculum do Dr. Eddy Giusepe Chirinos Isidro"
        )
    )
    server = ls.LitServer(api)
    server.run(port=8000)
