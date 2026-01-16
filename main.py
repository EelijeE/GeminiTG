from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from google import genai
from google.genai import types
import os

app = FastAPI()

# Получаем ключ
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Настраиваем НОВЫЙ клиент Google (по вашему примеру)
# Если ключа нет, клиент не создастся, ошибку обработаем позже
client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

# Список моделей для проверки (Новая библиотека любит простые имена)
MODELS_TO_TRY = [
    "gemini-2.0-flash",       # Самая новая стабильная
    "gemini-2.0-flash-lite",  # Облегченная версия 2.0
    "gemini-1.5-flash",       # Классика, работает всегда
    "gemini-1.5-pro",
    "gemini-3-flash-preview",
]

WORKING_MODEL = None

async def find_working_model():
    """Ищет рабочую модель, используя новый клиент"""
    global WORKING_MODEL
    if WORKING_MODEL: return WORKING_MODEL
    
    print("🔍 Тестируем модели через google.genai...")
    
    for model_name in MODELS_TO_TRY:
        try:
            # Тестовый запрос "Привет"
            response = client.models.generate_content(
                model=model_name,
                contents="Hi"
            )
            print(f"✅ УСПЕХ! Модель {model_name} работает!")
            WORKING_MODEL = model_name
            return model_name
        except Exception as e:
            error_str = str(e)
            # Если 429 (лимит) - идем дальше. Если 404 - идем дальше.
            print(f"❌ {model_name} не подошла: {error_str[:100]}...")
    
    return None

@app.post("/api/chat")
async def chat(request: Request):
    global WORKING_MODEL
    
    if not client:
        return JSONResponse({"reply": "⚠️ Ошибка: Нет API ключа в настройках Render."})

    # 1. Если модель еще не выбрана - ищем
    if not WORKING_MODEL:
        found = await find_working_model()
        if not found:
            return JSONResponse({"reply": "⚠️ Все модели заняты или недоступны. Проверьте квоты в Google AI Studio."})

    # 2. Обрабатываем сообщение пользователя
    data = await request.json()
    user_message = data.get("message", "")
    
    if not user_message:
        return JSONResponse({"error": "Пустое сообщение"})

    try:
        # 3. Отправляем запрос через НОВУЮ библиотеку
        response = client.models.generate_content(
            model=WORKING_MODEL,
            contents=user_message
        )
        
        # Достаем текст (в новой библиотеке это просто .text)
        return JSONResponse({"reply": response.text})

    except Exception as e:
        # Если рабочая модель вдруг отказала, сбрасываем выбор
        error_msg = str(e)
        if "429" in error_msg or "404" in error_msg:
            WORKING_MODEL = None
        return JSONResponse({"reply": f"Ошибка генерации: {error_msg}"})

# --- РАЗДАЧА САЙТА ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Загрузка...</h1>"

@app.get("/{filename}")
async def read_static(filename: str):
    if filename in ["script.js", "style.css"]:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                media_type = "application/javascript" if filename.endswith(".js") else "text/css"
                return HTMLResponse(content=f.read(), media_type=media_type)
    return JSONResponse({"error": "Not found"}, status_code=404)
