import os

from netra import Netra, NetraInstruments

from fastbrowse.config import settings


def init_observability() -> None:

    os.environ["NETRA_OTLP_ENDPOINT"] = settings.netra_otlp_endpoint
    
    Netra.init(
        headers={
            "x-api-key": settings.netra_api_key
        },
        environment="dev",
        app_name="fastbrowse",
        block_instruments={
            NetraInstruments.LANGCHAIN
        }
    )