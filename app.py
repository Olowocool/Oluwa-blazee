from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Basketball prediction API is running"}
