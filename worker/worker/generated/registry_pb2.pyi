import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RegisterRequest(_message.Message):
    __slots__ = ("worker_id", "address", "hardware")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    HARDWARE_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    address: str
    hardware: _common_pb2.HardwareInfo
    def __init__(self, worker_id: _Optional[str] = ..., address: _Optional[str] = ..., hardware: _Optional[_Union[_common_pb2.HardwareInfo, _Mapping]] = ...) -> None: ...

class RegisterResponse(_message.Message):
    __slots__ = ("accepted", "message", "server_time_unix_ms")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    SERVER_TIME_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    message: str
    server_time_unix_ms: int
    def __init__(self, accepted: _Optional[bool] = ..., message: _Optional[str] = ..., server_time_unix_ms: _Optional[int] = ...) -> None: ...

class HeartbeatRequest(_message.Message):
    __slots__ = ("worker_id", "client_time_unix_ms", "status", "cpu_utilization_percent", "ram_available_bytes", "ram_used_bytes", "ram_utilization_percent", "cpu_frequency_current_mhz", "cpu_load_average_1m", "swap_total_bytes", "swap_used_bytes", "swap_utilization_percent", "gpu_metrics", "metrics_collected_at_unix_ms")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    CLIENT_TIME_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CPU_UTILIZATION_PERCENT_FIELD_NUMBER: _ClassVar[int]
    RAM_AVAILABLE_BYTES_FIELD_NUMBER: _ClassVar[int]
    RAM_USED_BYTES_FIELD_NUMBER: _ClassVar[int]
    RAM_UTILIZATION_PERCENT_FIELD_NUMBER: _ClassVar[int]
    CPU_FREQUENCY_CURRENT_MHZ_FIELD_NUMBER: _ClassVar[int]
    CPU_LOAD_AVERAGE_1M_FIELD_NUMBER: _ClassVar[int]
    SWAP_TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    SWAP_USED_BYTES_FIELD_NUMBER: _ClassVar[int]
    SWAP_UTILIZATION_PERCENT_FIELD_NUMBER: _ClassVar[int]
    GPU_METRICS_FIELD_NUMBER: _ClassVar[int]
    METRICS_COLLECTED_AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    client_time_unix_ms: int
    status: str
    cpu_utilization_percent: float
    ram_available_bytes: int
    ram_used_bytes: int
    ram_utilization_percent: float
    cpu_frequency_current_mhz: float
    cpu_load_average_1m: float
    swap_total_bytes: int
    swap_used_bytes: int
    swap_utilization_percent: float
    gpu_metrics: _containers.RepeatedCompositeFieldContainer[_common_pb2.GPUMetrics]
    metrics_collected_at_unix_ms: int
    def __init__(self, worker_id: _Optional[str] = ..., client_time_unix_ms: _Optional[int] = ..., status: _Optional[str] = ..., cpu_utilization_percent: _Optional[float] = ..., ram_available_bytes: _Optional[int] = ..., ram_used_bytes: _Optional[int] = ..., ram_utilization_percent: _Optional[float] = ..., cpu_frequency_current_mhz: _Optional[float] = ..., cpu_load_average_1m: _Optional[float] = ..., swap_total_bytes: _Optional[int] = ..., swap_used_bytes: _Optional[int] = ..., swap_utilization_percent: _Optional[float] = ..., gpu_metrics: _Optional[_Iterable[_Union[_common_pb2.GPUMetrics, _Mapping]]] = ..., metrics_collected_at_unix_ms: _Optional[int] = ...) -> None: ...

class HeartbeatResponse(_message.Message):
    __slots__ = ("acknowledged", "server_time_unix_ms", "message", "reregister_required")
    ACKNOWLEDGED_FIELD_NUMBER: _ClassVar[int]
    SERVER_TIME_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    REREGISTER_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    acknowledged: bool
    server_time_unix_ms: int
    message: str
    reregister_required: bool
    def __init__(self, acknowledged: _Optional[bool] = ..., server_time_unix_ms: _Optional[int] = ..., message: _Optional[str] = ..., reregister_required: _Optional[bool] = ...) -> None: ...

class DeregisterRequest(_message.Message):
    __slots__ = ("worker_id", "reason")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    reason: str
    def __init__(self, worker_id: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class DeregisterResponse(_message.Message):
    __slots__ = ("accepted",)
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    def __init__(self, accepted: _Optional[bool] = ...) -> None: ...
