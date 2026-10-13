from functools import lru_cache

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from fastbrowse.config import settings


@lru_cache
def llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=SecretStr(settings.openai_api_key),
        base_url=settings.openai_base_url,
        model=settings.openai_model
    )