#!/bin/bash
exec /opt/rocm/core-10.0/bin/rocprofv3 --kernel-trace --stats -d /tmp/claude-1000/-home-dan/0413856d-bd18-464f-9674-36b76541259f/scratchpad/gpuprof -o run --output-format csv -- /home/dan/clone-of-d/src/Strata/build-hip/strata "$@"
