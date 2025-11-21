from fastapi import FastAPI
from . import upload 

app = FastAPI(title='Flowe classifier API') 

@app.get("/") 
def get_post(): 
    print("Flower Classifier running...")

app.include_router(router=upload.router)