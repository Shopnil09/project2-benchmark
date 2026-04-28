# Triton Inference Server with ResNet-50 baked in
# Lives at: cloudrun/Dockerfile

FROM nvcr.io/nvidia/tritonserver:24.01-py3

# Bake the model repository into the image
COPY model_repository /models

# Triton's HTTP, gRPC, and metrics ports
EXPOSE 8000 8001 8002

# Cloud Run will set $PORT to 8080 by default; we override to 8000 in deploy flags
# Triton listens on 8000 (HTTP) by default.
CMD ["tritonserver", \
     "--model-repository=/models", \
     "--strict-model-config=true", \
     "--log-verbose=1", \
     "--allow-metrics=true", \
     "--allow-http=true"]