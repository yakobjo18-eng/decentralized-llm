"""Generated protobuf/gRPC code (from shared/proto/*.proto).

Do not hand-edit anything in this directory -- regenerate it instead via
scripts/generate_protos.ps1 (or scripts/generate_protos.sh) whenever a
.proto file under shared/proto/ changes.

grpc_tools.protoc emits these files with flat, same-directory imports
(e.g. "import common_pb2 as common__pb2" inside registry_pb2.py) rather
than Python package-relative imports. To make those resolve correctly
when this package is imported normally as ``worker.generated``, this
directory is added to sys.path once, here, before anything else in the
package is imported.
"""

import os
import sys

_GENERATED_DIR = os.path.dirname(os.path.abspath(__file__))
if _GENERATED_DIR not in sys.path:
    sys.path.insert(0, _GENERATED_DIR)
