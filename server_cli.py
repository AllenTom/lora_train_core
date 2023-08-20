import os
from typing import Annotated

from fastapi import FastAPI, Request
from pydantic import Field, BaseModel
from server import project

app = FastAPI()


@app.get("/info")
def info():
    return {"message": "Hello World"}


class NewProjectBody(BaseModel):
    name: str = Field(
        default="None"
    )
    width: int = Field(
        default=512
    )
    height: int = Field(
        default=512
    )


@app.post("/action/newproject")
async def new_project(request: Request):
    data = await request.json()
    print(data)
    out = project.new_project(name=data["name"], width=data["width"], height=data["height"])
    return {
        "result": "success",
        "data":out
    }
