"""
Kubeflow Pipelines (KFP v2) definition for the VortexVision MLOps pipeline.
Covers: data prep → feature extraction → training → evaluation → export → deploy.

Compile with:
  python mlops/kubeflow_pipeline.py
  # Outputs: vortexvision_pipeline.yaml
"""
from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output


# ── Component definitions ─────────────────────────────────────────────────────

@dsl.component(base_image="vortexvision/mlops:latest", packages_to_install=["opencv-python-headless"])
def prepare_data_op(raw_data_path: str, processed_data: Output[Dataset]):
    import subprocess
    subprocess.run([
        "python", "mlops/prepare_data.py",
        "--raw-dir", raw_data_path,
        "--out-dir", processed_data.path,
    ], check=True)


@dsl.component(base_image="vortexvision/mlops:latest")
def extract_features_op(processed_data: Input[Dataset], sequences: Output[Dataset]):
    import subprocess
    subprocess.run([
        "python", "mlops/extract_features.py",
        "--data-dir", f"{processed_data.path}/train",
        "--out-dir", f"{sequences.path}/train",
    ], check=True)
    subprocess.run([
        "python", "mlops/extract_features.py",
        "--data-dir", f"{processed_data.path}/val",
        "--out-dir", f"{sequences.path}/val",
    ], check=True)


@dsl.component(base_image="vortexvision/mlops:latest", packages_to_install=["torch", "mlflow"])
def train_autoencoder_op(
    processed_data: Input[Dataset],
    epochs: int,
    autoencoder_model: Output[Model],
):
    import subprocess
    subprocess.run([
        "python", "mlops/train.py",
        "--data-dir", f"{processed_data.path}/normal_frames",
        "--epochs", str(epochs),
    ], check=True)
    import shutil
    shutil.copy("models/autoencoder_best.pt", autoencoder_model.path)


@dsl.component(base_image="vortexvision/mlops:latest", packages_to_install=["torch", "mlflow"])
def train_transformer_op(
    sequences: Input[Dataset],
    epochs: int,
    transformer_model: Output[Model],
):
    import subprocess
    subprocess.run([
        "python", "mlops/train_transformer.py",
        "--data-dir", f"{sequences.path}/train",
        "--epochs", str(epochs),
    ], check=True)
    import shutil
    shutil.copy("models/transformer_best.pt", transformer_model.path)


@dsl.component(base_image="vortexvision/mlops:latest")
def evaluate_op(
    processed_data: Input[Dataset],
    autoencoder_model: Input[Model],
    transformer_model: Input[Model],
    auc_roc: Output[float],
    f1_score: Output[float],
):
    import json, shutil
    shutil.copy(autoencoder_model.path, "models/autoencoder_best.pt")
    shutil.copy(transformer_model.path, "models/transformer_best.pt")
    import subprocess
    subprocess.run([
        "python", "mlops/evaluate.py",
        "--model-dir", "models",
        "--data-dir", f"{processed_data.path}/test",
    ], check=True)
    with open("metrics/eval_results.json") as f:
        metrics = json.load(f)
    auc_roc.set(metrics["auc_roc"])
    f1_score.set(metrics["f1_score"])


@dsl.component(base_image="vortexvision/mlops:latest")
def export_models_op(
    autoencoder_model: Input[Model],
    transformer_model: Input[Model],
    exported_models: Output[Dataset],
):
    import shutil
    shutil.copy(autoencoder_model.path, "models/autoencoder_best.pt")
    shutil.copy(transformer_model.path, "models/transformer_best.pt")
    import subprocess
    subprocess.run([
        "python", "mlops/model_export.py",
        "--model-dir", "models",
        "--out-dir", exported_models.path,
    ], check=True)


@dsl.component(base_image="vortexvision/mlops:latest")
def deploy_op(exported_models: Input[Dataset], auc_roc: float, min_auc: float = 0.80):
    """Deploy to KServe only if AUC-ROC meets the quality gate."""
    if auc_roc < min_auc:
        raise ValueError(f"AUC-ROC {auc_roc:.3f} below threshold {min_auc}. Deployment blocked.")
    import subprocess
    subprocess.run([
        "kubectl", "apply", "-f", "infra/k8s/base/kserve-inferenceservice.yaml"
    ], check=True)
    print(f"Deployed successfully. AUC-ROC: {auc_roc:.3f}")


# ── Pipeline definition ───────────────────────────────────────────────────────

@dsl.pipeline(
    name="VortexVision Training Pipeline",
    description="End-to-end MLOps pipeline: data prep → training → evaluation → deployment",
)
def vortexvision_pipeline(
    raw_data_path: str = "s3://vortexvision-data/raw",
    ae_epochs: int = 50,
    tf_epochs: int = 30,
    min_auc_roc: float = 0.80,
):
    prepare = prepare_data_op(raw_data_path=raw_data_path)

    extract = extract_features_op(processed_data=prepare.outputs["processed_data"])

    train_ae = train_autoencoder_op(
        processed_data=prepare.outputs["processed_data"],
        epochs=ae_epochs,
    )

    train_tf = train_transformer_op(
        sequences=extract.outputs["sequences"],
        epochs=tf_epochs,
    )

    evaluate = evaluate_op(
        processed_data=prepare.outputs["processed_data"],
        autoencoder_model=train_ae.outputs["autoencoder_model"],
        transformer_model=train_tf.outputs["transformer_model"],
    )

    export = export_models_op(
        autoencoder_model=train_ae.outputs["autoencoder_model"],
        transformer_model=train_tf.outputs["transformer_model"],
    )
    export.after(evaluate)

    deploy = deploy_op(
        exported_models=export.outputs["exported_models"],
        auc_roc=evaluate.outputs["auc_roc"],
        min_auc=min_auc_roc,
    )


if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(vortexvision_pipeline, "vortexvision_pipeline.yaml")
    print("Pipeline compiled → vortexvision_pipeline.yaml")
