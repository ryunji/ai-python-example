from fastapi import FastAPI, Form
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")
app = FastAPI()

@app.post("/text/")
async def text(text: str = Form()):
    result = classifier(text)
    return {"result": result}