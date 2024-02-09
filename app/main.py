from fastapi import FastAPI

from . import items

app = FastAPI()


@app.get("/")
async def read_root() -> dict[str, str]:
    return {"Hello": "World"}


app.include_router(items.router)
