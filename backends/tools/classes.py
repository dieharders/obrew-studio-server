from enum import Enum


class TOOL_SCHEMA_TYPES(str, Enum):
    JSON = "json"
    TYPESCRIPT = "typescript"


DEFAULT_TOOL_SCHEMA_TYPE = TOOL_SCHEMA_TYPES.JSON.value
