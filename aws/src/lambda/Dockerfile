# Lambda ARM64 image for Cardio ML inference.
#
# Build context: PROJECT ROOT (not this directory).
# CDK handles this automatically via DockerImageCode.fromImageAsset.

FROM public.ecr.aws/lambda/python:3.12-arm64

COPY pyproject.toml README.md ${LAMBDA_TASK_ROOT}/
COPY src/ ${LAMBDA_TASK_ROOT}/src/

# Install runtime dependencies only + mangum adapter.
# --no-deps on cardio-ml avoids pulling dev/evidently deps.
# Pin sklearn to match the version used during training (joblib serialization
# is version-sensitive for sklearn estimators).
RUN pip install --no-cache-dir \
    "scikit-learn==1.5.2" \
    xgboost \
    pandas \
    numpy \
    fastapi \
    pydantic \
    joblib \
    psutil \
    mangum \
    && pip install --no-cache-dir --no-deps -e ${LAMBDA_TASK_ROOT}/

COPY aws/src/lambda/model/ ${LAMBDA_TASK_ROOT}/model/
COPY aws/src/lambda/handler.py ${LAMBDA_TASK_ROOT}/

ENV MODEL_PATH=${LAMBDA_TASK_ROOT}/model/pipeline.joblib \
    MODEL_VERSION=1 \
    CARDIO_N_JOBS=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

CMD ["handler.handler"]
