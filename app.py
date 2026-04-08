from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"status": "running"}

@app.get("/reset")
def reset():
    return {"status": "reset successful"}