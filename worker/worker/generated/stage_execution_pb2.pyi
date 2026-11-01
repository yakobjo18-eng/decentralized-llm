from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ProtocolVersion(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PROTOCOL_VERSION_UNSPECIFIED: _ClassVar[ProtocolVersion]
    PROTOCOL_VERSION_1: _ClassVar[ProtocolVersion]

class TensorDType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TENSOR_DTYPE_UNSPECIFIED: _ClassVar[TensorDType]
    TENSOR_DTYPE_FLOAT32: _ClassVar[TensorDType]
    TENSOR_DTYPE_INT64: _ClassVar[TensorDType]

class ActivationKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTIVATION_KIND_UNSPECIFIED: _ClassVar[ActivationKind]
    ACTIVATION_KIND_TOKEN_IDS: _ClassVar[ActivationKind]
    ACTIVATION_KIND_HIDDEN_STATES: _ClassVar[ActivationKind]
    ACTIVATION_KIND_LOGITS: _ClassVar[ActivationKind]

class StageStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STAGE_STATUS_UNSPECIFIED: _ClassVar[StageStatus]
    STAGE_STATUS_OK: _ClassVar[StageStatus]
    STAGE_STATUS_REJECTED: _ClassVar[StageStatus]
    STAGE_STATUS_BUSY: _ClassVar[StageStatus]
    STAGE_STATUS_FAILED: _ClassVar[StageStatus]
    STAGE_STATUS_DEADLINE_EXCEEDED: _ClassVar[StageStatus]
PROTOCOL_VERSION_UNSPECIFIED: ProtocolVersion
PROTOCOL_VERSION_1: ProtocolVersion
TENSOR_DTYPE_UNSPECIFIED: TensorDType
TENSOR_DTYPE_FLOAT32: TensorDType
TENSOR_DTYPE_INT64: TensorDType
ACTIVATION_KIND_UNSPECIFIED: ActivationKind
ACTIVATION_KIND_TOKEN_IDS: ActivationKind
ACTIVATION_KIND_HIDDEN_STATES: ActivationKind
ACTIVATION_KIND_LOGITS: ActivationKind
STAGE_STATUS_UNSPECIFIED: StageStatus
STAGE_STATUS_OK: StageStatus
STAGE_STATUS_REJECTED: StageStatus
STAGE_STATUS_BUSY: StageStatus
STAGE_STATUS_FAILED: StageStatus
STAGE_STATUS_DEADLINE_EXCEEDED: StageStatus

class TensorPayload(_message.Message):
    __slots__ = ("encoding", "dtype", "shape", "data", "kind", "produced_by_stage", "positions_start")
    ENCODING_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    PRODUCED_BY_STAGE_FIELD_NUMBER: _ClassVar[int]
    POSITIONS_START_FIELD_NUMBER: _ClassVar[int]
    encoding: str
    dtype: TensorDType
    shape: _containers.RepeatedScalarFieldContainer[int]
    data: bytes
    kind: ActivationKind
    produced_by_stage: int
    positions_start: int
    def __init__(self, encoding: _Optional[str] = ..., dtype: _Optional[_Union[TensorDType, str]] = ..., shape: _Optional[_Iterable[int]] = ..., data: _Optional[bytes] = ..., kind: _Optional[_Union[ActivationKind, str]] = ..., produced_by_stage: _Optional[int] = ..., positions_start: _Optional[int] = ...) -> None: ...

class ExecuteStageRequest(_message.Message):
    __slots__ = ("protocol_version", "request_id", "plan_id", "plan_json", "model_id", "config_fingerprint", "target_worker_id", "stage_index", "unit_start", "unit_end", "input", "coordinator")
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PLAN_ID_FIELD_NUMBER: _ClassVar[int]
    PLAN_JSON_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FINGERPRINT_FIELD_NUMBER: _ClassVar[int]
    TARGET_WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    STAGE_INDEX_FIELD_NUMBER: _ClassVar[int]
    UNIT_START_FIELD_NUMBER: _ClassVar[int]
    UNIT_END_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    COORDINATOR_FIELD_NUMBER: _ClassVar[int]
    protocol_version: ProtocolVersion
    request_id: str
    plan_id: str
    plan_json: bytes
    model_id: str
    config_fingerprint: str
    target_worker_id: str
    stage_index: int
    unit_start: int
    unit_end: int
    input: TensorPayload
    coordinator: str
    def __init__(self, protocol_version: _Optional[_Union[ProtocolVersion, str]] = ..., request_id: _Optional[str] = ..., plan_id: _Optional[str] = ..., plan_json: _Optional[bytes] = ..., model_id: _Optional[str] = ..., config_fingerprint: _Optional[str] = ..., target_worker_id: _Optional[str] = ..., stage_index: _Optional[int] = ..., unit_start: _Optional[int] = ..., unit_end: _Optional[int] = ..., input: _Optional[_Union[TensorPayload, _Mapping]] = ..., coordinator: _Optional[str] = ...) -> None: ...

class StageError(_message.Message):
    __slots__ = ("code", "message", "problems", "executed")
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    PROBLEMS_FIELD_NUMBER: _ClassVar[int]
    EXECUTED_FIELD_NUMBER: _ClassVar[int]
    code: str
    message: str
    problems: _containers.RepeatedScalarFieldContainer[str]
    executed: bool
    def __init__(self, code: _Optional[str] = ..., message: _Optional[str] = ..., problems: _Optional[_Iterable[str]] = ..., executed: _Optional[bool] = ...) -> None: ...

class StageTiming(_message.Message):
    __slots__ = ("validate_seconds", "compute_seconds", "total_seconds")
    VALIDATE_SECONDS_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_SECONDS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_SECONDS_FIELD_NUMBER: _ClassVar[int]
    validate_seconds: float
    compute_seconds: float
    total_seconds: float
    def __init__(self, validate_seconds: _Optional[float] = ..., compute_seconds: _Optional[float] = ..., total_seconds: _Optional[float] = ...) -> None: ...

class ExecuteStageResponse(_message.Message):
    __slots__ = ("protocol_version", "request_id", "worker_id", "stage_index", "status", "output", "timing", "error", "worker_hostname", "worker_pid")
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    STAGE_INDEX_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    TIMING_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    WORKER_HOSTNAME_FIELD_NUMBER: _ClassVar[int]
    WORKER_PID_FIELD_NUMBER: _ClassVar[int]
    protocol_version: ProtocolVersion
    request_id: str
    worker_id: str
    stage_index: int
    status: StageStatus
    output: TensorPayload
    timing: StageTiming
    error: StageError
    worker_hostname: str
    worker_pid: int
    def __init__(self, protocol_version: _Optional[_Union[ProtocolVersion, str]] = ..., request_id: _Optional[str] = ..., worker_id: _Optional[str] = ..., stage_index: _Optional[int] = ..., status: _Optional[_Union[StageStatus, str]] = ..., output: _Optional[_Union[TensorPayload, _Mapping]] = ..., timing: _Optional[_Union[StageTiming, _Mapping]] = ..., error: _Optional[_Union[StageError, _Mapping]] = ..., worker_hostname: _Optional[str] = ..., worker_pid: _Optional[int] = ...) -> None: ...
