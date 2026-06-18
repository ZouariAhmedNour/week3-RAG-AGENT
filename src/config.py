import os
from dotenv import load_dotenv
from google import genai


def get_env(name: str, default: str | None = None) -> str:
    load_dotenv()

    value = os.getenv(name, default)

    if value is None:
        raise EnvironmentError(f"Variable d'environnement manquante : {name}")

    return value


def get_gemini_client():
    api_key = get_env("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)