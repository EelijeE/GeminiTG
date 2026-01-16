from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import google.generativeai as genai
import os

app = FastAPI()

# Получаем ключ из ЛЮБОЙ переменной (старой или новой)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("MY_SECRET_KEY")

# Настройка модели
def get_gemini_response(text):
    if not GEMINI_API_KEY:
        return "⚠️ Ошибка: Нет API ключа. Проверьте настройки сервера."
    
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        # Используем самую стабильную модель
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(text)
        return response.text
    except Exception as e:
        return f"Ошибка Google API: {str(e)}"

# Эндпоинт для чата
@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    if not user_message:
        return JSONResponse({"error": "Пустое сообщение"})
    
    bot_reply = get_gemini_response(user_message)
    return JSONResponse({"reply": bot_reply})

# Эндпоинт для главной страницы
@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Если файл index.html есть - отдаем его
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Ошибка: файл index.html не найден</h1>"

# Раздача скрипта и стилей (если они лежат рядом)
@app.get("/{filename}")
async def read_static(filename: str):
    if filename in ["script.js", "style.css"]:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                media_type = "application/javascript" if filename.endswith(".js") else "text/css"
                return HTMLResponse(content=f.read(), media_type=media_type)
    return JSONResponse({"error": "Not found"}, status_code=404)
