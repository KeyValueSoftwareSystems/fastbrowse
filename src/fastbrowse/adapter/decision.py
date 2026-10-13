from functools import lru_cache

from langchain_typesafe import TypeSafeClassifier

from fastbrowse.config import settings


@lru_cache
def decision() -> TypeSafeClassifier:
    return TypeSafeClassifier(
        api_key=settings.typesafe_api_key,
        base_url=settings.typesafe_base_url,
        model=settings.typesafe_model
    )