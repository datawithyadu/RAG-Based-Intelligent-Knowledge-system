"""FastAPI entry point for the RAG knowledge system."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="RAG Knowledge System")

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


class Question(BaseModel):
    text: str
    top_k: int = 4


class Answer(BaseModel):
    question: str
    answer: str
    sources: list[str]


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"title": "Ask my documents"})


@app.post("/ask", response_model=Answer)
def ask(question: Question) -> Answer:
    # TODO: call your retrieval + generation services here
    return Answer(
        question=question.text,
        answer=f"(placeholder answer, top_k={question.top_k})",
        sources=[],
    )


def main() -> None:
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()