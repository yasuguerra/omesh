import openai
import logging
from .settings import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def get_openai_response(prompt, context=""):
    """Get a response from OpenAI."""
    try:
        full_prompt = f"{context}\n\nPregunta/Tarea: {prompt}\n\nRespond in English:"
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are Omesh AI assistant specializing in web and social analytics. Respond in clear and concise English focused on actionable insights."},
                {"role": "user", "content": full_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error llamando a OpenAI: {e}")
        return f"Hubo un error al contactar al asistente de IA: {e}. ¿Está bien configurada la API Key?"