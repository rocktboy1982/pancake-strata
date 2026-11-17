@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%CUDA_PATH%\bin;%PATH%"
cd /d "C:\Users\iancu\build-latest\src"
cmake -S . -B "C:\Users\iancu\build-latest\build-sm75" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
if errorlevel 1 exit /b 1
set RC=0
for %%T in (strata-poly poly_workers_test poly_cpu_kernels_test poly_mxfp4_test poly_route_test poly_gdn_cols_test poly_gemm_small_test poly_attend_split_test poly_expert_list_test poly_dsv4_test poly_kv_q4_test poly_expert_parity poly_cpu_experts_bench poly_draft_race_test head_argmax_parity head_gather_parity head_fmt_test q5k_head_test) do ( cmake --build "C:\Users\iancu\build-latest\build-sm75" -j 4 --target %%T || ( echo BUILD-FAIL %%T & set RC=1 ) )
exit /b %RC%
