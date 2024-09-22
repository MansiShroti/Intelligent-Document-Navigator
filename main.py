from typing import Union
from typing import Annotated
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer
import torch
from fastapi.middleware.cors import CORSMiddleware
import os


from helper import uploader, query

app = FastAPI()
# app.mount("/static", StaticFiles(directory="static",html = True), name="static")

templates = Jinja2Templates(directory=os.getenv("STATIC"))
# templates = Jinja2Templates(directory="/opt/source-code/static")

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    print(f"You are using {device}. This is much slower than using "
        "a CUDA-enabled GPU. If on Colab you can change this by "
        "clicking Runtime > Change runtime type > GPU.")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

@app.post("/upload/",response_class=HTMLResponse)
async def create_upload_files(request:Request,files: list[UploadFile]):
    ans=[]
    title="The following files were uploaded or already exists"
    print(files)
    if len(files) == 0 or files[0].filename=='':
        return  templates.TemplateResponse("alert.html",{"request":request,"message":"No files were selected. Please select atleast a file to upload!!"})

    for file in files:
        message = await uploader(file,model)
        ans.append(message)
       
    return templates.TemplateResponse("uploaded.html",{"request":request,"ans":ans,"title":title})

@app.get("/{item_id}",response_class=HTMLResponse)
async def read_item(request:Request,item_id: str, q: Union[str, None] = None,th: Union[str,None]=None):
    if q:
        if th=='' or th is None:
            threshold = 0.4
        else:
            threshold = float(th)

        ans = await query(q,model,threshold)
        title="These are the results of the semantic search for threshhold = "+str(threshold)
        return templates.TemplateResponse("uploaded.html",{"request":request,"ans":ans,"title":title})
    return  templates.TemplateResponse("alert.html",{"request":request,"message":"No query was sent"})

@app.get("/",response_class=HTMLResponse)
async def main(request:Request):
    return templates.TemplateResponse("index.html", {"request": request})
    # return FileResponse("index.html")