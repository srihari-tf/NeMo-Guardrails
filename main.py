import json
import os
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import Body, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from pydantic import BaseModel, Field


class ContentType(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None


class CitationSource(BaseModel):
    startIndex: Optional[int] = None
    endIndex: Optional[int] = None
    uri: Optional[str] = None
    license: Optional[str] = None


class JsonSchema(BaseModel):
    # This is a placeholder for any JSON object
    __root__: Dict[str, Any]


class Function(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[JsonSchema] = None


class Tool(BaseModel):
    type: str
    function: Optional[Function] = None


class CitationMetadata(BaseModel):
    citationSources: Optional[List[CitationSource]] = None


class ContentType(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None


class CitationSource(BaseModel):
    startIndex: Optional[int] = None
    endIndex: Optional[int] = None
    uri: Optional[str] = None
    license: Optional[str] = None


class CitationMetadata(BaseModel):
    citationSources: Optional[List[CitationSource]] = None


class Message(BaseModel):
    role: Union[Literal["system"], Literal["user"],
                Literal["assistant"], Literal["function"], Literal["tool"]]
    content: Optional[Union[str, List[ContentType]]] = None
    name: Optional[str] = None
    function_call: Optional[Any] = Field(
        default=None, description="Arbitrary JSON object or any value representing a function call.")
    tool_calls: Optional[Any] = Field(
        default=None, description="Arbitrary JSON object or any value representing tool calls.")
    tool_call_id: Optional[str] = None
    citationMetadata: Optional[CitationMetadata] = None


class Content(BaseModel):
    role: Union[Literal["system"], Literal["user"],
                Literal["assistant"], Literal["function"], Literal["tool"]]
    content: Optional[Union[str, List[ContentType]]] = None
    name: Optional[str] = None
    function_call: Optional[Any] = None
    tool_calls: Optional[Any] = None
    tool_call_id: Optional[str] = None
    citationMetadata: Optional[CitationMetadata] = None


class Example(BaseModel):
    input: Optional[Content] = None
    output: Optional[Content] = None


class InputRailsConfig(BaseModel):
    name: str


class Params(BaseModel):
    model: str
    prompt: Optional[Union[str, List[str]]]
    messages: Optional[List[Message]]
    functions: Optional[List[Function]]
    function_call: Optional[Union[str, Dict[str, str]]]
    max_tokens: Optional[int] = 100
    temperature: Optional[float]
    top_p: Optional[float]
    n: Optional[int]
    stream: Optional[bool]
    logprobs: Optional[int]
    echo: Optional[bool]
    stop: Optional[Union[str, List[str]]]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
    best_of: Optional[int]
    logit_bias: Optional[Dict[str, int]]
    user: Optional[str]
    context: Optional[str]
    examples: Optional[List[Example]]
    top_k: Optional[int]
    tools: Optional[List[Tool]]
    tool_choice: Optional[str]
    tfy_log_request: Optional[bool] = None
    custom_metadata: Optional[dict] = None
    rails_config: Optional[InputRailsConfig]


app = FastAPI()

# endpoint to list all guardrails with name
@app.get("/guardrails")
async def list_guardrails():
    list_of_guardrails: List[InputRailsConfig]  = [];
    for guardrail in os.listdir("./config"):
        try:
            RailsConfig.from_path(f"./config/{guardrail}")
            list_of_guardrails.append(InputRailsConfig(name=guardrail))
        except:
            pass
    
    return {"guardrails": list_of_guardrails}

@app.post("/chat/completions")
async def chat_completions(request: Params):
    config = RailsConfig(models=[],passthrough=True)
    if request.rails_config:
        config_path = f"./config/{request.rails_config.name}"
        try:
            config = RailsConfig.from_path(config_path)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"Config '{request.config_name}' not found.")

    guardrails = RunnableRails(config)

    # Construct extra_headers if necessary
    extra_headers = {}
    if request.tfy_log_request is not None:
        extra_headers["X-TFY-METADATA"] = json.dumps({
            "tfy_log_request": request.tfy_log_request and str(request.tfy_log_request).lower() or "false",
            **(request.custom_metadata if request.custom_metadata else {})
        })

    llm = ChatOpenAI(
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        streaming=False,
        api_key="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6ImxzV0lDNWtkU1V1bXg1ckg5NkR6bFdYUGxJTSJ9.eyJhdWQiOiI4OTUyNTNhZi1lYzlkLTRiZTYtODNkMS02ZjI0OGU2NDRlNzkiLCJleHAiOjM3MTE1Mjc2NjEsImlhdCI6MTcxMTUyNzY2MSwiaXNzIjoidHJ1ZWZvdW5kcnkuY29tIiwic3ViIjoiY2x1OWpkbXhzMDE4bzAxbzY0b2ZtMWYzNSIsImp0aSI6IjYxNDYyZmVkLTA4NjgtNGQyNS05Njg4LTYzMDg5MThhODYxMCIsInVzZXJuYW1lIjoibXktbGFuZ2NoYWluLWFwaS1rZXkiLCJ1c2VyVHlwZSI6InNlcnZpY2VhY2NvdW50IiwidGVuYW50TmFtZSI6InRydWVmb3VuZHJ5Iiwicm9sZXMiOlsidGVuYW50LWFkbWluIiwidGVuYW50LW1lbWJlciJdLCJhcHBsaWNhdGlvbklkIjoiODk1MjUzYWYtZWM5ZC00YmU2LTgzZDEtNmYyNDhlNjQ0ZTc5In0.Klf9Y4xur0dwVUB7sKTBIsdQo2_9u9cYQ2a-jxKO89ijjIxf6hwzZS2zOCED3YCpD45kT0jqOYlpj4GzY7UKyhqOzsdw9ob0kHaaINXLqiLJtiL0Hm8OOkg23RAuE2-HJIElr0Xlc2_hXab9cbruRE_oVbLZfB4dBcHi0AdC4cXgay7yuP98DTe1ARjt88XLwkay53psvye7iWoc0knxCQ8FIhVp6j-c2eJMkKB1qiPHQZ778MXclGr_4ER8sX6CwgGvC2THFQZvuerJc3xHLUv_r-OByfybO6C1bchVAvFx4RYY3Jd6_C_cCzDk13PAcQ64iG9O4CU33Aew4e1Iwg",
        base_url="https://app.devtest.truefoundry.tech/api/llm/openai",
        extra_headers=extra_headers,

    )

    prompt = ChatPromptTemplate.from_messages(
        [(message.role, message.content) for message in request.messages])
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    chain_with_guardrails = guardrails | chain

    response = await run_in_threadpool(chain_with_guardrails.invoke, {"input": "why do you think?"})
    return {"response": response}


@app.post("/completions")
async def completions(request: Params):
    config = RailsConfig(models=[],passthrough=True)
    if request.rails_config:
        config_path = f"./config/{request.rails_config.name}"
        try:
            config = RailsConfig.from_path(config_path)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"Config '{request.config_name}' not found.")

    guardrails = RunnableRails(config)

    # Construct extra_headers if necessary
    extra_headers = {}
    if request.tfy_log_request is not None:
        extra_headers["X-TFY-METADATA"] = json.dumps({
            "tfy_log_request": request.tfy_log_request and str(request.tfy_log_request).lower() or "false",
            **(request.custom_metadata if request.custom_metadata else {})
        })

    llm = OpenAI(
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        streaming=False,
        api_key="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6ImxzV0lDNWtkU1V1bXg1ckg5NkR6bFdYUGxJTSJ9.eyJhdWQiOiI4OTUyNTNhZi1lYzlkLTRiZTYtODNkMS02ZjI0OGU2NDRlNzkiLCJleHAiOjM3MTE1Mjc2NjEsImlhdCI6MTcxMTUyNzY2MSwiaXNzIjoidHJ1ZWZvdW5kcnkuY29tIiwic3ViIjoiY2x1OWpkbXhzMDE4bzAxbzY0b2ZtMWYzNSIsImp0aSI6IjYxNDYyZmVkLTA4NjgtNGQyNS05Njg4LTYzMDg5MThhODYxMCIsInVzZXJuYW1lIjoibXktbGFuZ2NoYWluLWFwaS1rZXkiLCJ1c2VyVHlwZSI6InNlcnZpY2VhY2NvdW50IiwidGVuYW50TmFtZSI6InRydWVmb3VuZHJ5Iiwicm9sZXMiOlsidGVuYW50LWFkbWluIiwidGVuYW50LW1lbWJlciJdLCJhcHBsaWNhdGlvbklkIjoiODk1MjUzYWYtZWM5ZC00YmU2LTgzZDEtNmYyNDhlNjQ0ZTc5In0.Klf9Y4xur0dwVUB7sKTBIsdQo2_9u9cYQ2a-jxKO89ijjIxf6hwzZS2zOCED3YCpD45kT0jqOYlpj4GzY7UKyhqOzsdw9ob0kHaaINXLqiLJtiL0Hm8OOkg23RAuE2-HJIElr0Xlc2_hXab9cbruRE_oVbLZfB4dBcHi0AdC4cXgay7yuP98DTe1ARjt88XLwkay53psvye7iWoc0knxCQ8FIhVp6j-c2eJMkKB1qiPHQZ778MXclGr_4ER8sX6CwgGvC2THFQZvuerJc3xHLUv_r-OByfybO6C1bchVAvFx4RYY3Jd6_C_cCzDk13PAcQ64iG9O4CU33Aew4e1Iwg",
        base_url="https://app.devtest.truefoundry.tech/api/llm/openai",
        extra_headers=extra_headers,
    )

    prompt = PromptTemplate.from_template("{input}")
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    chain_with_guardrails = guardrails | chain

    response = await run_in_threadpool(chain_with_guardrails.invoke, {"input": request.prompt})
    return {"response": response}
