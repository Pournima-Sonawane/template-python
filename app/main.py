from typing import Any, Dict

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from . import example, items


class CustomFastAPI(FastAPI):
    def openapi(self) -> Dict[str, Any]:
        if self.openapi_schema:
            return self.openapi_schema
        openapi_schema = get_openapi(
            title="Template-python OpenAPI",
            version="0.1.0",
            description="This is a OpenAPI schema of the template-python app",
            routes=self.routes,
        )
        self.openapi_schema = openapi_schema
        return self.openapi_schema


app = CustomFastAPI()


app.include_router(example.router)
app.include_router(items.routes.router)
