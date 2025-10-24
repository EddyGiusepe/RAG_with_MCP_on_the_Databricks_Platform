#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Script server.py
================

Run
---
uv run server.py
"""
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
import os
from pydantic import BaseModel
from litserve.mcp import MCP
import litserve as ls
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# os.environ["OPENAI_API_KEY"] = os.environ.get("LIGHTNING_API_KEY", "")
# os.environ["OPENAI_API_BASE"] = "https://lightning.ai/v1"

model = "gpt-4.1-nano" #"gpt-4o"


class RequestType(BaseModel):
    query: str


class DocumentChatAPI(ls.LitAPI):
    def setup(self, device):
        Settings.llm = OpenAI(
            api_key=OPENAI_API_KEY,
            temperature=0.1,
            model=model,
        )
        documents = SimpleDirectoryReader("/home/eddygiusepe/2_GitHub/RAG_with_MCP_on_the_Databricks_Platform/data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        self.query_engine = index.as_query_engine()

    def decode_request(self, request: RequestType):
        return request.query

    def predict(self, query: str):
        return self.query_engine.query(query)

    def encode_response(self, output) -> dict:
        return {"output": output.response}


if __name__ == "__main__":
    api = DocumentChatAPI(mcp=MCP(description="Respostas a perguntas sobre o curriculum do Dr. Eddy Giusepe Chirinos Isidro"))
    server = ls.LitServer(api)
    server.run(port=8000)
