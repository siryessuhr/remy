from fastapi import FastAPI

app = FastAPI(title="Remy", version="0.1.0")


@app.get("/")
def healthcheck():
    return {"status": "ok"}
