import initapp

initapp.init_global()
from fastapi import FastAPI, Request, UploadFile, WebSocket
from pydantic import Field, BaseModel

from server import project, training, sender

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
        "data": out
    }


@app.post("/action/loadproject")
async def load_project(request: Request):
    data = await request.json()
    print(data)
    out = project.load_project_service(data["name"])
    return {
        "result": "success",
        "data": out
    }


@app.post("/action/addoriginal")
async def add_original(file: UploadFile, id: str):
    new_items = project.add_original_image(file, id)
    return {
        "result": "success",
        "data": new_items
    }


@app.post("/action/auto_preprocess")
async def create_upload_files(files: list[UploadFile], id: str):
    return project.auto_caption_image_service(id, files)


@app.post("/action/create_dataset")
async def create_dataset(request: Request):
    data = await request.json()
    return project.create_dataset(
        id=data["id"],
        name=data["name"],
        step=data["step"],
        image_hashes=data["hashes"],
    )


@app.post("/action/add_trainconfig")
async def add_trainconfig(request: Request):
    data = await request.json()
    return project.add_train_config(
        id=data["id"],
        name=data["name"],
        pretrained_model_name_or_path=data["pretrained_model_name_or_path"],
        model_name=data["model_name"],
        config_id=data.get("config_id", None),
        extra_params=data.get("extra_params", {}),
        lora_preset_name=data.get("lora_preset_name", "default"),
    )


@app.post("/action/train")
async def train(request: Request):
    data = await request.json()
    return training.train_project(
        id=data["id"],
        config_id=data["config_id"],
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    sender.active_connections.append(websocket)  # Add the connection to the list

    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except Exception as e:
        print(f"WebSocket connection closed: {e}")
    finally:
        sender.active_connections.remove(websocket)

@app.get("/action/out")
async def get_out():
    return training.out_text

@app.get("/action/getprojectmeta")
async def get_project_meta(id: str):
    meta = project.read_project_meta(id)
    return {
        "result": "success",
        "data": meta
    }