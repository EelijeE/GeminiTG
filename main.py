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

# СПИСОК СТАБИЛЬНЫХ МОДЕЛЕЙ (Текст + Фото)
MODELS_TO_TRY = [
    "gemini-3-flash-preview",# Мощная (резерв)
    "gemini-1.5-flash",          # Самая надежная
    "gemini-1.5-flash-latest",   # Свежая
    "gemini-2.0-flash-lite-preview-02-05", # Быстрая новая
    "gemini-2.0-flash"  
]

async def generate_with_fallback(contents):
    if not client: return "Ошибка: Нет ключа."
    
    last_error = ""
    
    for model_name in MODELS_TO_TRY:
        try:
            print(f"🔄 Пробую: {model_name}")
            response = client.models.generate_content(
                model=model_name,
                contents=contents
            )
            print(f"✅ Успех: {model_name}")
            return response.text
        except Exception as e:
            print(f"❌ Пропуск {model_name}: {e}")
            last_error = str(e)
            continue
            
    return f"Не удалось получить ответ. ({last_error[:100]})"

@app.post("/api/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message", "")
        image_b64 = data.get("image", None)

        parts = []

        # 1. КАРТИНКА
        if image_b64:
            try:
                image_bytes = base64.b64decode(image_b64)
                parts.append(
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                )
            except:
                pass

        # 2. ТЕКСТ
        if user_message:
            parts.append(types.Part.from_text(text=user_message))
        
        # 3. ЕСЛИ ТОЛЬКО ФОТО
        if not user_message and image_b64:
            parts.append(types.Part.from_text(text="Что на этом фото?"))

        if not parts:
            return JSONResponse({"reply": "Пустое сообщение"})

        reply_text = await generate_with_fallback(contents=[types.Content(parts=parts)])
        
        return JSONResponse({"reply": reply_text})

    except Exception as e:
        return JSONResponse({"reply": f"Ошибка сервера: {str(e)}"})

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
