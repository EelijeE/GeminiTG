from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from google import genai
from google.genai import types
import os
import base64
import traceback

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Клиент
client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except:
        pass

# СПИСОК МОДЕЛЕЙ (ОТ СТАБИЛЬНЫХ К НОВЫМ)
# Мы ставим 1.5 Flash первой, потому что у неё идеальная память.
MODELS_TO_TRY = [
    "gemini-1.5-flash",          # Самая надежная рабочая лошадка
    "gemini-2.0-flash",          # Новая, умная (если 1.5 не справится)
    "gemini-2.0-flash-lite-preview-02-05",
]

async def generate_with_fallback(contents):
    if not client: return "Ошибка: Нет ключа."
    
    last_error = ""
    
    # Настройки генерации (делаем бота чуть строже к фактам)
    config = types.GenerateContentConfig(
        temperature=0.7,
        system_instruction="Ты полезный и вежливый ассистент. Ты всегда помнишь контекст беседы и имя пользователя, если он представился."
    )

    for model_name in MODELS_TO_TRY:
        print(f"🔄 Пробую модель: {model_name}")
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config 
            )
            print(f"✅ Успех на модели: {model_name}")
            return response.text
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Ошибка {model_name}: {error_msg}")
            last_error = error_msg
            continue # Идем к следующей
    
    return f"Все модели заняты или недоступны. Ошибка: {last_error}"

@app.post("/api/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message", "")
        image_b64 = data.get("image", None)
        history = data.get("history", []) 

        # Собираем контекст для отправки
        contents = []

        # 1. ЗАГРУЖАЕМ ПРОШЛОЕ (ИСТОРИЮ)
        for msg in history:
            role = msg.get("role") 
            text = msg.get("text")
            if text:
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=text)]
                ))

        # 2. ДОБАВЛЯЕМ ТЕКУЩЕЕ СООБЩЕНИЕ
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

        if current_parts:
            contents.append(types.Content(role="user", parts=current_parts))
        else:
             return JSONResponse({"reply": "Пустой запрос"})

        # 3. ГЕНЕРАЦИЯ
        reply_text = await generate_with_fallback(contents=contents)
        
        return JSONResponse({"reply": reply_text})

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return JSONResponse({"reply": f"Ошибка сервера: {str(e)}"})

# --- Раздача файлов ---
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
