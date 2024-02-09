# This file uses Classy-FastAPI to create a FastAPI
# instance with dependency-injection of an *instance* of some class.
# This is especially handy if you want to maintain some common resource reused between
# invocations of the routes, e.g. a database connection.
# For detailed explanation and more examples see:
# https://gitlab.com/companionlabs-opensource/classy-fastapi

from typing import Union

from classy_fastapi import Routable, delete, get
from fastapi import FastAPI


class KVStore:
    def __init__(self):
        self.kvs: dict[str, int] = {"abc": 123}

    def get_value(self, key: str) -> Union[int | None]:
        return self.kvs.get(key)

    def delete_key(self, key: str) -> None:
        self.kvs.pop(key)


class KVSRoutes(Routable):
    def __init__(self, kvstore: KVStore) -> None:
        super().__init__()
        self.__kvs = kvstore

    @get("/key/{key}")
    def get_value(self, key: str) -> Union[int | None]:
        return self.__kvs.get_value(key)

    @delete("/key/{key}")
    def delete_key(self, key: str) -> None:
        self.__kvs.delete_key(key)


kvs_instance = KVStore()

# Simple intuitive injection
kvs_routes_instance = KVSRoutes(kvs_instance)

app = FastAPI()
app.include_router(kvs_routes_instance.router)
