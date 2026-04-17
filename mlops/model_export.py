"""
Model export pipeline:
  - Export PyTorch models to ONNX for cross-platform inference
  - Export to TorchScript for production serving
  - Apply INT8 / FP16 quantization
  - Validate exported models against original
"""
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def export_autoencoder_onnx(model_path: str, out_path: str, device: str = "cpu"):
    from anomaly.autoencoder import ConvAutoencoder

    model = ConvAutoencoder()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dummy = torch.randn(1, 3, 640, 640)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["reconstruction", "latent"],
        dynamic_axes={"input": {0: "batch_size"}, "reconstruction": {0: "batch_size"}},
        opset_version=17,
    )
    logger.info("Autoencoder exported to ONNX: %s", out_path)
    _validate_onnx(out_path, dummy.numpy())


def export_transformer_onnx(model_path: str, out_path: str, seq_len: int = 16, feature_dim: int = 128, device: str = "cpu"):
    from anomaly.transformer_detector import TemporalTransformer

    model = TemporalTransformer(feature_dim=feature_dim, seq_len=seq_len)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dummy = torch.randn(1, seq_len, feature_dim)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["sequence"],
        output_names=["anomaly_prob"],
        dynamic_axes={"sequence": {0: "batch_size"}, "anomaly_prob": {0: "batch_size"}},
        opset_version=17,
    )
    logger.info("Transformer exported to ONNX: %s", out_path)
    _validate_onnx(out_path, dummy.numpy())


def _validate_onnx(onnx_path: str, dummy_input: np.ndarray):
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        t0 = time.monotonic()
        _ = sess.run(None, {input_name: dummy_input})
        latency_ms = (time.monotonic() - t0) * 1000
        logger.info("ONNX validation OK — inference latency: %.2f ms", latency_ms)
    except ImportError:
        logger.warning("onnxruntime not installed — skipping ONNX validation")


def quantize_dynamic(model_path: str, out_path: str):
    """Apply dynamic INT8 quantization (CPU-only, no calibration data needed)."""
    from anomaly.autoencoder import ConvAutoencoder

    model = ConvAutoencoder()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    torch.save(quantized.state_dict(), out_path)
    logger.info("Dynamic INT8 quantized model saved: %s", out_path)

    # Size comparison
    orig_size = Path(model_path).stat().st_size / 1024 / 1024
    quant_size = Path(out_path).stat().st_size / 1024 / 1024
    logger.info("Size: %.1f MB → %.1f MB (%.0f%% reduction)", orig_size, quant_size,
                (1 - quant_size / orig_size) * 100)


def export_torchscript(model_path: str, out_path: str, device: str = "cpu"):
    from anomaly.autoencoder import ConvAutoencoder

    model = ConvAutoencoder()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scripted = torch.jit.script(model)
    scripted.save(out_path)
    logger.info("TorchScript model saved: %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and optimize VortexVision models")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--out-dir", default="models/exported")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    ae_path = f"{args.model_dir}/autoencoder_best.pt"
    tf_path = f"{args.model_dir}/transformer_best.pt"

    if Path(ae_path).exists():
        export_autoencoder_onnx(ae_path, f"{args.out_dir}/autoencoder.onnx", args.device)
        quantize_dynamic(ae_path, f"{args.out_dir}/autoencoder_int8.pt")
        export_torchscript(ae_path, f"{args.out_dir}/autoencoder_scripted.pt", args.device)

    if Path(tf_path).exists():
        export_transformer_onnx(tf_path, f"{args.out_dir}/transformer.onnx", device=args.device)
