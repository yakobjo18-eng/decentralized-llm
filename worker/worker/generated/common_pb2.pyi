from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DATA_TYPE_UNSPECIFIED: _ClassVar[DataType]
    DATA_TYPE_FLOAT32: _ClassVar[DataType]
    DATA_TYPE_FLOAT16: _ClassVar[DataType]
    DATA_TYPE_BFLOAT16: _ClassVar[DataType]
    DATA_TYPE_INT64: _ClassVar[DataType]
    DATA_TYPE_INT32: _ClassVar[DataType]
    DATA_TYPE_BOOL: _ClassVar[DataType]

class ProfileSource(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PROFILE_SOURCE_UNSPECIFIED: _ClassVar[ProfileSource]
    PROFILE_SOURCE_REAL: _ClassVar[ProfileSource]
    PROFILE_SOURCE_SIMULATED: _ClassVar[ProfileSource]
DATA_TYPE_UNSPECIFIED: DataType
DATA_TYPE_FLOAT32: DataType
DATA_TYPE_FLOAT16: DataType
DATA_TYPE_BFLOAT16: DataType
DATA_TYPE_INT64: DataType
DATA_TYPE_INT32: DataType
DATA_TYPE_BOOL: DataType
PROFILE_SOURCE_UNSPECIFIED: ProfileSource
PROFILE_SOURCE_REAL: ProfileSource
PROFILE_SOURCE_SIMULATED: ProfileSource

class GPUInfo(_message.Message):
    __slots__ = ("vendor", "name", "memory_total_bytes", "memory_type", "backends", "driver_version", "id", "dedicated_memory_bytes", "shared_memory_bytes", "memory_source", "integrated")
    VENDOR_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    MEMORY_TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_TYPE_FIELD_NUMBER: _ClassVar[int]
    BACKENDS_FIELD_NUMBER: _ClassVar[int]
    DRIVER_VERSION_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    DEDICATED_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    SHARED_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_SOURCE_FIELD_NUMBER: _ClassVar[int]
    INTEGRATED_FIELD_NUMBER: _ClassVar[int]
    vendor: str
    name: str
    memory_total_bytes: int
    memory_type: str
    backends: _containers.RepeatedScalarFieldContainer[str]
    driver_version: str
    id: str
    dedicated_memory_bytes: int
    shared_memory_bytes: int
    memory_source: str
    integrated: bool
    def __init__(self, vendor: _Optional[str] = ..., name: _Optional[str] = ..., memory_total_bytes: _Optional[int] = ..., memory_type: _Optional[str] = ..., backends: _Optional[_Iterable[str]] = ..., driver_version: _Optional[str] = ..., id: _Optional[str] = ..., dedicated_memory_bytes: _Optional[int] = ..., shared_memory_bytes: _Optional[int] = ..., memory_source: _Optional[str] = ..., integrated: _Optional[bool] = ...) -> None: ...

class GPUMetrics(_message.Message):
    __slots__ = ("gpu_id", "utilization_percent", "dedicated_memory_used_bytes", "shared_memory_used_bytes")
    GPU_ID_FIELD_NUMBER: _ClassVar[int]
    UTILIZATION_PERCENT_FIELD_NUMBER: _ClassVar[int]
    DEDICATED_MEMORY_USED_BYTES_FIELD_NUMBER: _ClassVar[int]
    SHARED_MEMORY_USED_BYTES_FIELD_NUMBER: _ClassVar[int]
    gpu_id: str
    utilization_percent: float
    dedicated_memory_used_bytes: int
    shared_memory_used_bytes: int
    def __init__(self, gpu_id: _Optional[str] = ..., utilization_percent: _Optional[float] = ..., dedicated_memory_used_bytes: _Optional[int] = ..., shared_memory_used_bytes: _Optional[int] = ...) -> None: ...

class BackendCapability(_message.Message):
    __slots__ = ("name", "available", "device_count", "version", "devices", "tested", "test_passed", "test_gflops", "detail")
    NAME_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_COUNT_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    DEVICES_FIELD_NUMBER: _ClassVar[int]
    TESTED_FIELD_NUMBER: _ClassVar[int]
    TEST_PASSED_FIELD_NUMBER: _ClassVar[int]
    TEST_GFLOPS_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    name: str
    available: bool
    device_count: int
    version: str
    devices: _containers.RepeatedScalarFieldContainer[str]
    tested: bool
    test_passed: bool
    test_gflops: float
    detail: str
    def __init__(self, name: _Optional[str] = ..., available: _Optional[bool] = ..., device_count: _Optional[int] = ..., version: _Optional[str] = ..., devices: _Optional[_Iterable[str]] = ..., tested: _Optional[bool] = ..., test_passed: _Optional[bool] = ..., test_gflops: _Optional[float] = ..., detail: _Optional[str] = ...) -> None: ...

class HardwareInfo(_message.Message):
    __slots__ = ("hostname", "cpu_logical_cores", "total_ram_bytes", "has_cuda_gpu", "gpu_name", "gpu_vram_bytes", "os", "python_version", "hardware_schema_version", "cpu_manufacturer", "cpu_model", "cpu_physical_cores", "cpu_architecture", "os_name", "os_version", "os_build", "gpus", "torch_installed", "torch_version", "cpu_backend_available", "cuda_available", "cuda_device_count", "directml_available", "cpu_utilization_percent", "ram_available_bytes", "ram_used_bytes", "ram_utilization_percent", "profile_source", "profile_collected_at_unix_ms", "simulated_profile_name", "cpu_frequency_current_mhz", "cpu_frequency_max_mhz", "cpu_load_average_1m", "swap_total_bytes", "swap_used_bytes", "swap_utilization_percent", "backend_capabilities", "torch_cuda_build_version", "gpu_metrics", "backend_probe_pending")
    HOSTNAME_FIELD_NUMBER: _ClassVar[int]
    CPU_LOGICAL_CORES_FIELD_NUMBER: _ClassVar[int]
    TOTAL_RAM_BYTES_FIELD_NUMBER: _ClassVar[int]
    HAS_CUDA_GPU_FIELD_NUMBER: _ClassVar[int]
    GPU_NAME_FIELD_NUMBER: _ClassVar[int]
    GPU_VRAM_BYTES_FIELD_NUMBER: _ClassVar[int]
    OS_FIELD_NUMBER: _ClassVar[int]
    PYTHON_VERSION_FIELD_NUMBER: _ClassVar[int]
    HARDWARE_SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    CPU_MANUFACTURER_FIELD_NUMBER: _ClassVar[int]
    CPU_MODEL_FIELD_NUMBER: _ClassVar[int]
    CPU_PHYSICAL_CORES_FIELD_NUMBER: _ClassVar[int]
    CPU_ARCHITECTURE_FIELD_NUMBER: _ClassVar[int]
    OS_NAME_FIELD_NUMBER: _ClassVar[int]
    OS_VERSION_FIELD_NUMBER: _ClassVar[int]
    OS_BUILD_FIELD_NUMBER: _ClassVar[int]
    GPUS_FIELD_NUMBER: _ClassVar[int]
    TORCH_INSTALLED_FIELD_NUMBER: _ClassVar[int]
    TORCH_VERSION_FIELD_NUMBER: _ClassVar[int]
    CPU_BACKEND_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    CUDA_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    CUDA_DEVICE_COUNT_FIELD_NUMBER: _ClassVar[int]
    DIRECTML_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    CPU_UTILIZATION_PERCENT_FIELD_NUMBER: _ClassVar[int]
    RAM_AVAILABLE_BYTES_FIELD_NUMBER: _ClassVar[int]
    RAM_USED_BYTES_FIELD_NUMBER: _ClassVar[int]
    RAM_UTILIZATION_PERCENT_FIELD_NUMBER: _ClassVar[int]
    PROFILE_SOURCE_FIELD_NUMBER: _ClassVar[int]
    PROFILE_COLLECTED_AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    SIMULATED_PROFILE_NAME_FIELD_NUMBER: _ClassVar[int]
    CPU_FREQUENCY_CURRENT_MHZ_FIELD_NUMBER: _ClassVar[int]
    CPU_FREQUENCY_MAX_MHZ_FIELD_NUMBER: _ClassVar[int]
    CPU_LOAD_AVERAGE_1M_FIELD_NUMBER: _ClassVar[int]
    SWAP_TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    SWAP_USED_BYTES_FIELD_NUMBER: _ClassVar[int]
    SWAP_UTILIZATION_PERCENT_FIELD_NUMBER: _ClassVar[int]
    BACKEND_CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    TORCH_CUDA_BUILD_VERSION_FIELD_NUMBER: _ClassVar[int]
    GPU_METRICS_FIELD_NUMBER: _ClassVar[int]
    BACKEND_PROBE_PENDING_FIELD_NUMBER: _ClassVar[int]
    hostname: str
    cpu_logical_cores: int
    total_ram_bytes: int
    has_cuda_gpu: bool
    gpu_name: str
    gpu_vram_bytes: int
    os: str
    python_version: str
    hardware_schema_version: int
    cpu_manufacturer: str
    cpu_model: str
    cpu_physical_cores: int
    cpu_architecture: str
    os_name: str
    os_version: str
    os_build: str
    gpus: _containers.RepeatedCompositeFieldContainer[GPUInfo]
    torch_installed: bool
    torch_version: str
    cpu_backend_available: bool
    cuda_available: bool
    cuda_device_count: int
    directml_available: bool
    cpu_utilization_percent: float
    ram_available_bytes: int
    ram_used_bytes: int
    ram_utilization_percent: float
    profile_source: ProfileSource
    profile_collected_at_unix_ms: int
    simulated_profile_name: str
    cpu_frequency_current_mhz: float
    cpu_frequency_max_mhz: float
    cpu_load_average_1m: float
    swap_total_bytes: int
    swap_used_bytes: int
    swap_utilization_percent: float
    backend_capabilities: _containers.RepeatedCompositeFieldContainer[BackendCapability]
    torch_cuda_build_version: str
    gpu_metrics: _containers.RepeatedCompositeFieldContainer[GPUMetrics]
    backend_probe_pending: bool
    def __init__(self, hostname: _Optional[str] = ..., cpu_logical_cores: _Optional[int] = ..., total_ram_bytes: _Optional[int] = ..., has_cuda_gpu: _Optional[bool] = ..., gpu_name: _Optional[str] = ..., gpu_vram_bytes: _Optional[int] = ..., os: _Optional[str] = ..., python_version: _Optional[str] = ..., hardware_schema_version: _Optional[int] = ..., cpu_manufacturer: _Optional[str] = ..., cpu_model: _Optional[str] = ..., cpu_physical_cores: _Optional[int] = ..., cpu_architecture: _Optional[str] = ..., os_name: _Optional[str] = ..., os_version: _Optional[str] = ..., os_build: _Optional[str] = ..., gpus: _Optional[_Iterable[_Union[GPUInfo, _Mapping]]] = ..., torch_installed: _Optional[bool] = ..., torch_version: _Optional[str] = ..., cpu_backend_available: _Optional[bool] = ..., cuda_available: _Optional[bool] = ..., cuda_device_count: _Optional[int] = ..., directml_available: _Optional[bool] = ..., cpu_utilization_percent: _Optional[float] = ..., ram_available_bytes: _Optional[int] = ..., ram_used_bytes: _Optional[int] = ..., ram_utilization_percent: _Optional[float] = ..., profile_source: _Optional[_Union[ProfileSource, str]] = ..., profile_collected_at_unix_ms: _Optional[int] = ..., simulated_profile_name: _Optional[str] = ..., cpu_frequency_current_mhz: _Optional[float] = ..., cpu_frequency_max_mhz: _Optional[float] = ..., cpu_load_average_1m: _Optional[float] = ..., swap_total_bytes: _Optional[int] = ..., swap_used_bytes: _Optional[int] = ..., swap_utilization_percent: _Optional[float] = ..., backend_capabilities: _Optional[_Iterable[_Union[BackendCapability, _Mapping]]] = ..., torch_cuda_build_version: _Optional[str] = ..., gpu_metrics: _Optional[_Iterable[_Union[GPUMetrics, _Mapping]]] = ..., backend_probe_pending: _Optional[bool] = ...) -> None: ...
