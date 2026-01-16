from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from google import genai
from google.genai import types
import os
import base64

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Клиент Google
client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

# Список моделей (2.0 Flash отлично видит картинки!)
MODEL_NAME = "gemini-3-flash-preview" 

@app.post("/api/chat")
async def chat(request: Request):
    if not client:
        return JSONResponse({"reply": "⚠️ Ошибка: Нет API ключа."})

    data = await request.json()
    user_message = data.get("message", "")
    image_b64 = data.get("image", None) # Получаем картинку

    # Собираем содержимое для отправки
    contents = []
    
    # 1. Если есть картинка - добавляем её
    if image_b64:
        try:
            image_bytes = base64.b64decode(image_b64)
            contents.append(
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            )
        except Exception as e:
            return JSONResponse({"reply": f"Ошибка обработки картинки: {str(e)}"})

    # 2. Если есть текст - добавляем его (или просим описать фото, если текста нет)
    if user_message:
        contents.append(types.Part.from_text(text=user_message))
    elif image_b64:
        contents.append(types.Part.from_text(text="Что изображено на этой картинке? Подробно опиши."))
    else:
        return JSONResponse({"reply": "Пришлите текст или фото."})

    try:
        # Отправляем мультимодальный запрос
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Content(parts=contents)]
        )
        return JSONResponse({"reply": response.text})

    except Exception as e:
        return JSONResponse({"reply": f"Ошибка Gemini: {str(e)}"})

# --- Раздача файлов сайта ---
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
