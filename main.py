from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from google import genai
from google.genai import types
import os
import base64
import traceback

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Клиент Google
client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Ошибка создания клиента: {e}")

# СПИСОК МОДЕЛЕЙ (По приоритету)
# Мы поставили 3-ю версию первой, так как вы сказали, что она работала
MODELS_TO_TRY = [
    "gemini-3-flash-preview", 
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite-preview-02-05", # Свежая версия из вашего списка
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

async def generate_with_fallback(contents):
    """Пытается отправить запрос по очереди во все модели"""
    if not client:
        return "Ошибка: Не настроен API ключ."

    last_error = ""
    
    for model_name in MODELS_TO_TRY:
        print(f"🔄 Пробую модель: {model_name}...")
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents
            )
            print(f"✅ УСПЕХ! Сработала модель: {model_name}")
            return response.text
        except Exception as e:
            error_str = str(e)
            print(f"❌ {model_name} не справилась: {error_str}")
            last_error = error_str
            # Если ошибка критическая (нет доступа), пробуем следующую
            continue
    
    return f"Все модели дали сбой. Последняя ошибка: {last_error}"

@app.post("/api/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message", "")
        image_b64 = data.get("image", None)

        # Собираем части сообщения (текст + картинка)
        parts = []
        
        # 1. Обработка картинки
        if image_b64:
            try:
                # Декодируем base64 в байты
                image_bytes = base64.b64decode(image_b64)
                parts.append(
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                )
            except Exception as e:
                print(f"Ошибка картинки: {e}")
                return JSONResponse({"reply": "Не удалось обработать картинку."})

        # 2. Обработка текста
        if user_message:
            parts.append(types.Part.from_text(text=user_message))
        elif image_b64:
            # Если текста нет, но есть фото - добавляем промпт
            parts.append(types.Part.from_text(text="Опиши подробно, что на этом изображении?"))
        else:
            return JSONResponse({"reply": "Пустое сообщение."})

        # 3. Запуск генерации с перебором моделей
        # Формируем объект Content правильно для новой библиотеки
        content_obj = types.Content(parts=parts)
        
        reply_text = await generate_with_fallback(contents=[content_obj])
        
        return JSONResponse({"reply": reply_text})

    except Exception as e:
        # Ловим любые падения сервера, чтобы не было ошибки 502
        error_trace = traceback.format_exc()
        print(f"🔥 КРИТИЧЕСКАЯ ОШИБКА СЕРВЕРА:\n{error_trace}")
        return JSONResponse({"reply": f"Ошибка сервера (см. логи): {str(e)}"})

# --- Раздача статики ---
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
