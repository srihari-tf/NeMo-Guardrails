import json
import os
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import Body, FastAPI, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from pydantic import BaseModel, Field
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

load_dotenv() 
BASE_URL = os.getenv("BASE_URL", "https://app.devtest.truefoundry.tech/api/llm/openai")

# Model definitions
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
    role: Union[Literal["system"], Literal["user"], Literal["assistant"], Literal["function"], Literal["tool"]]
    content: Optional[Union[str, List[ContentType]]] = None
    name: Optional[str] = None
    function_call: Optional[Any] = Field(default=None, description="Arbitrary JSON object or any value representing a function call.")
    tool_calls: Optional[Any] = Field(default=None, description="Arbitrary JSON object or any value representing tool calls.")
    tool_call_id: Optional[str] = None
    citationMetadata: Optional[CitationMetadata] = None

class Function(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class Tool(BaseModel):
    type: str
    function: Optional[Function] = None

class Example(BaseModel):
    input: Optional[Message] = None
    output: Optional[Message] = None

class InputRailsConfig(BaseModel):
    name: str

class Params(BaseModel):
    model: str
    prompt: Optional[Union[str, List[str]]] = None
    messages: Optional[List[Message]] = None
    functions: Optional[List[Function]] = None
    function_call: Optional[Union[str, Dict[str, str]]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    logprobs: Optional[int] = None
    echo: Optional[bool] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    best_of: Optional[int] = None
    logit_bias: Optional[Dict[str, int]] = None
    user: Optional[str] = None
    context: Optional[str] = None
    examples: Optional[List[Example]] = None
    top_k: Optional[int] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[str] = None
    tfy_log_request: Optional[bool] = None
    custom_metadata: Optional[dict] = None
    rails_config: Optional[InputRailsConfig] = None


def get_model_kwargs_from_params(params: Params) -> Dict[str, Any]:
    model_kwargs = {}
    for key in params.model_fields.keys():
        if key in ["model", "prompt", "messages", "rails_config", "tfy_log_request", "custom_metadata", "logit_bias", "temperature"]:
            continue
        value = getattr(params, key)
        if value is not None:
            model_kwargs[key] = value
    return model_kwargs

app = FastAPI()
security = HTTPBearer()

def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    return credentials.credentials

@app.get("/guardrails")
async def list_guardrails():
    list_of_guardrails: List[InputRailsConfig] = []
    for guardrail in os.listdir("./config"):
        try:
            RailsConfig.from_path(f"./config/{guardrail}")
            list_of_guardrails.append(InputRailsConfig(name=guardrail))
        except:
            pass

    return {"guardrails": list_of_guardrails}


@app.post("/chat/completions")
async def chat_completions(request: Params, api_key: str = Depends(get_api_key)):
    config = RailsConfig(models=[], passthrough=True)
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
        model_kwargs=get_model_kwargs_from_params(request),
        streaming=False,
        logit_bias=request.logit_bias,
        api_key=api_key,
        base_url=BASE_URL,
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
async def completions(request: Params, api_key: str = Depends(get_api_key)):
    config = RailsConfig(models=[], passthrough=True)
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
    
    print("get_model_kwargs_from_params(request),", get_model_kwargs_from_params(request),)
    llm = OpenAI(
        model=request.model,
        model_kwargs=get_model_kwargs_from_params(request),
        streaming=False,
        temperature=request.temperature,
        logit_bias=request.logit_bias,
        api_key=api_key,
        base_url=BASE_URL,
        extra_headers=extra_headers,
    )

    prompt = PromptTemplate.from_template("{input}")
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    chain_with_guardrails = guardrails | chain

    response = await run_in_threadpool(chain_with_guardrails.invoke, {"input": request.prompt})
    return {"response": response}
