from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from google import genai
from google.genai import types
import os
import base64
import traceback

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except:
        pass

# Список моделей
MODELS_TO_TRY = [
    "gemini-3-flash-preview", 
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite-preview-02-05",
    "gemini-1.5-flash"
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
            continue # Пробуем следующую
    
    return "Не удалось получить ответ от нейросети."

@app.post("/api/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message", "")
        image_b64 = data.get("image", None)
        history = data.get("history", []) # <-- Получаем историю от браузера

        # Собираем ВЕСЬ диалог (contents)
        contents = []

        # 1. Сначала добавляем старые сообщения в список
        for msg in history:
            role = msg.get("role") # user или model
            text = msg.get("text")
            if text:
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=text)]
                ))

        # 2. Теперь готовим ТЕКУЩЕЕ сообщение
        current_parts = []
        
        if image_b64:
            try:
                image_bytes = base64.b64decode(image_b64)
                current_parts.append(
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                )
            except:
                pass

        if user_message:
            current_parts.append(types.Part.from_text(text=user_message))
        elif image_b64:
            current_parts.append(types.Part.from_text(text="Что на этом изображении?"))

        # Добавляем текущее сообщение в конец списка
        if current_parts:
            contents.append(types.Content(role="user", parts=current_parts))
        else:
             return JSONResponse({"reply": "Пустой запрос"})

        # 3. Отправляем ВЕСЬ список
        reply_text = await generate_with_fallback(contents=contents)
        
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
