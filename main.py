from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from google import genai
from google.genai import types
import os
import base64

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except:
        pass

# Для аудио лучше всего подходит 1.5 Flash (она быстрая и мультимодальная)
MODELS_TO_TRY = [
    "gemini-1.5-flash", 
    "gemini-1.5-flash-latest",
    "gemini-2.0-flash"
]

async def generate_with_fallback(contents):
    if not client: return "Ошибка: Нет ключа."
    
    for model_name in MODELS_TO_TRY:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents
            )
            return response.text
        except Exception as e:
            print(f"Ошибка {model_name}: {e}")
            continue
    return "Не удалось обработать запрос."

@app.post("/api/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message", "")
        image_b64 = data.get("image", None)
        audio_b64 = data.get("audio", None) # <-- Получаем аудио

        parts = []

        # 1. Если есть АУДИО
        if audio_b64:
            try:
                audio_bytes = base64.b64decode(audio_b64)
                # Браузеры обычно пишут в webm или ogg. Gemini 1.5 это понимает.
                parts.append(
                    types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg")
                )
            except Exception as e:
                return JSONResponse({"reply": f"Ошибка аудио: {e}"})

        # 2. Если есть ФОТО
        if image_b64:
            try:
                image_bytes = base64.b64decode(image_b64)
                parts.append(
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                )
            except:
                pass

        # 3. Если есть ТЕКСТ
        if user_message:
            parts.append(types.Part.from_text(text=user_message))
        
        # 4. Если нет текста, но есть медиа - добавляем промпт
        if not user_message and (image_b64 or audio_b64):
            if audio_b64 and image_b64:
                parts.append(types.Part.from_text(text="Послушай аудио и посмотри на картинку. Что ты думаешь?"))
            elif audio_b64:
                parts.append(types.Part.from_text(text="Что сказано в этом аудио? Ответь на том же языке."))
            elif image_b64:
                parts.append(types.Part.from_text(text="Что на фото?"))

        if not parts:
            return JSONResponse({"reply": "Пустой запрос"})

        # Генерация
        reply_text = await generate_with_fallback(contents=[types.Content(parts=parts)])
        
        return JSONResponse({"reply": reply_text})

    except Exception as e:
        return JSONResponse({"reply": f"Ошибка: {str(e)}"})

# --- Статика ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f: return f.read()
    return "Загрузка..."

@app.get("/{filename}")
async def read_static(filename: str):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f: return f.read()
    return JSONResponse({"error": "Not found"}, status_code=404)
