from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import requests
import os
import json

app = FastAPI()

# 1. Получаем ключ
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# 2. Список моделей для теста (от самых новых к старым)
# Мы взяли gemini-3-flash-preview из твоего примера
CANDIDATE_MODELS = [
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-1.5-flash-002",
    "gemini-pro"
]

# Глобальная переменная для хранения работающей модели
WORKING_MODEL = None

def test_model_connection(model_name):
    """Проверяет, работает ли конкретная модель через прямой запрос"""
    if not GEMINI_API_KEY:
        return False, "Нет API ключа"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": "Hello"}]}]
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            return True, model_name
        else:
            return False, f"Ошибка {response.status_code}: {response.text}"
    except Exception as e:
        return False, str(e)

def find_working_model():
    """Перебирает список и ищет живую модель"""
    global WORKING_MODEL
    
    # Если уже нашли раньше - не ищем снова
    if WORKING_MODEL:
        return WORKING_MODEL

    print("🔍 Начинаю поиск рабочей модели...")
    
    for model in CANDIDATE_MODELS:
        print(f"Testing {model}...")
        success, result = test_model_connection(model)
        if success:
            WORKING_MODEL = model
            print(f"✅ УСПЕХ! Выбрана модель: {model}")
            return model
        else:
            print(f"❌ {model} не работает. Причина: {result}")
    
    return None

# --- API ЭНДПОИНТЫ ---

@app.post("/api/chat")
async def chat(request: Request):
    global WORKING_MODEL
    
    # 1. Если модель еще не выбрана - ищем её сейчас
    if not WORKING_MODEL:
        found = find_working_model()
        if not found:
            return JSONResponse({"reply": "⚠️ Не удалось найти ни одной рабочей модели. Проверьте логи Render."})

    # 2. Подготовка запроса к выбранной модели
    data = await request.json()
    user_message = data.get("message", "")
    
    if not user_message:
        return JSONResponse({"error": "Empty message"})

    # 3. Прямой запрос (CURL-style)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{WORKING_MODEL}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": user_message}]}]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            # Если вдруг рабочая модель начала сбоить (429 или 500)
            return JSONResponse({"reply": f"Ошибка модели ({WORKING_MODEL}): {response.text}"})
            
        result_json = response.json()
        
        # Парсим ответ Google
        try:
            bot_text = result_json['candidates'][0]['content']['parts'][0]['text']
            return JSONResponse({"reply": bot_text})
        except (KeyError, IndexError):
            return JSONResponse({"reply": "Модель прислала пустой ответ."})

    except Exception as e:
        return JSONResponse({"reply": f"Ошибка сети: {str(e)}"})

# --- РАЗДАЧА СТАТИКИ (САЙТ) ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Сайт загружается... Обновите страницу через минуту.</h1>"

@app.get("/{filename}")
async def read_static(filename: str):
    if filename in ["script.js", "style.css"]:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                media_type = "application/javascript" if filename.endswith(".js") else "text/css"
                return HTMLResponse(content=f.read(), media_type=media_type)
    return JSONResponse({"error": "Not found"}, status_code=404)
