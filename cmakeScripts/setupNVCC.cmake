execute_process(
    COMMAND bash -c "nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.'"
    OUTPUT_VARIABLE GPU_COMPUTE_CAPS_LOCAL
  )
  MESSAGE(STATUS "Detected gpu capabilities: ${GPU_COMPUTE_CAPS_LOCAL}")
  set(CMAKE_CUDA_ARCHITECTURES "${GPU_COMPUTE_CAPS_LOCAL}")