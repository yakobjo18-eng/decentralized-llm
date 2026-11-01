from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class HealthCheckRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("status", "worker_id", "server_time_unix_ms")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    SERVER_TIME_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    status: str
    worker_id: str
    server_time_unix_ms: int
    def __init__(self, status: _Optional[str] = ..., worker_id: _Optional[str] = ..., server_time_unix_ms: _Optional[int] = ...) -> None: ...
