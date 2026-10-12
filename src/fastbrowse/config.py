from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = ""

    # Typesafe
    typesafe_api_key: str = ""
    typesafe_base_url: str = ""
    typesafe_model: str = ""


settings = Settings()