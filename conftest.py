"""
Root conftest.py — installs mock modules for heavy dependencies
before any test collection happens. This runs before tests/conftest.py.
"""
import sys
from unittest.mock import MagicMock


def _mock(name: str, **attrs):
    """Create a MagicMock module and register it in sys.modules."""
    mod = MagicMock()
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# ── Mock heavy ML/infra deps that aren't installed in CI ─────────────────────

# torch — provide just enough for anomaly models to import
torch_mock = _mock("torch")
torch_mock.Tensor = MagicMock
torch_mock.nn = MagicMock()
torch_mock.no_grad = lambda: (lambda f: f)  # decorator passthrough
torch_mock.load = MagicMock(return_value={})
torch_mock.save = MagicMock()
torch_mock.tensor = MagicMock()
torch_mock.arange = MagicMock()
torch_mock.stack = MagicMock()
torch_mock.zeros = MagicMock()
torch_mock.ones = MagicMock()
torch_mock.randn = MagicMock()
torch_mock.rand = MagicMock()
torch_mock.float32 = "float32"
torch_mock.qint8 = "qint8"

_mock("torch.nn")
_mock("torch.nn.functional")
_mock("torch.utils")
_mock("torch.utils.data")
_mock("torch.optim")
_mock("torch.optim.lr_scheduler")
_mock("torch.quantization")
_mock("torch.jit")
_mock("torchvision")
_mock("torchvision.transforms")
_mock("torchvision.transforms.functional")

# ultralytics
_mock("ultralytics")
_mock("ultralytics.YOLO")

# confluent_kafka
confluent_mock = _mock("confluent_kafka")
confluent_mock.Producer = MagicMock
confluent_mock.Consumer = MagicMock
confluent_mock.KafkaError = MagicMock()
confluent_mock.KafkaError._PARTITION_EOF = -191

# cv2 — provide imencode/imdecode stubs
import numpy as np  # noqa: E402

cv2_mock = _mock("cv2")
cv2_mock.imencode = MagicMock(return_value=(True, np.zeros(100, dtype=np.uint8)))
cv2_mock.imdecode = MagicMock(return_value=np.zeros((640, 640, 3), dtype=np.uint8))
cv2_mock.resize = MagicMock(side_effect=lambda img, size, **kw: img)
cv2_mock.cvtColor = MagicMock(side_effect=lambda img, code: img)
cv2_mock.VideoCapture = MagicMock
cv2_mock.VideoWriter = MagicMock
cv2_mock.VideoWriter_fourcc = MagicMock(return_value=0)
cv2_mock.rectangle = MagicMock()
cv2_mock.putText = MagicMock()
cv2_mock.getTextSize = MagicMock(return_value=((50, 15), 0))
cv2_mock.circle = MagicMock()
cv2_mock.line = MagicMock()
cv2_mock.addWeighted = MagicMock()
cv2_mock.IMWRITE_JPEG_QUALITY = 1
cv2_mock.FONT_HERSHEY_SIMPLEX = 0
cv2_mock.CAP_PROP_BUFFERSIZE = 38
cv2_mock.COLOR_BGR2RGB = 4
cv2_mock.COLOR_BGR2GRAY = 6

# ray
_mock("ray")
_mock("ray.serve")

# mlflow
_mock("mlflow")
_mock("mlflow.pytorch")

# evidently
_mock("evidently")
_mock("evidently.report")
_mock("evidently.metric_preset")
_mock("evidently.pipeline")
_mock("evidently.pipeline.column_mapping")

# opentelemetry
_mock("opentelemetry")
_mock("opentelemetry.trace")
_mock("opentelemetry.sdk")
_mock("opentelemetry.sdk.trace")
_mock("opentelemetry.sdk.trace.export")
_mock("opentelemetry.sdk.resources")
_mock("opentelemetry.exporter")
_mock("opentelemetry.exporter.otlp")
_mock("opentelemetry.exporter.otlp.proto")
_mock("opentelemetry.exporter.otlp.proto.grpc")
_mock("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")
_mock("opentelemetry.instrumentation")
_mock("opentelemetry.instrumentation.fastapi")
_mock("opentelemetry.instrumentation.sqlalchemy")

# kfp
_mock("kfp")
_mock("kfp.dsl")

# PIL
_mock("PIL")
_mock("PIL.Image")

# yt_dlp
_mock("yt_dlp")
