from typing import Any, Dict
import logging

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from prometheus_fastapi_instrumentator import Instrumentator


from . import dell_uc2_opentraj 

app = FastAPI(title="OpenTraj Workload API")

Instrumentator().instrument(app).expose(app)


@app.on_event("startup")
async def run_workload_on_startup():
    logging.info("Starting OpenTraj Workload ....")
    dell_uc2_opentraj.run_workload()


@app.get("/")
def read_root():
    return {"message": "OpenTraj Workload API is running"}


