# parallel-video-motion-detection
This project implements a high-performance video motion detection system utilizing frame-differencing algorithms and a dynamic background model. The primary objective is to evaluate and compare the scalability, speedup, and efficiency of various parallel computing paradigms.

---

# Video Object Detection Pipeline - Performance Benchmark

A high-performance computer vision pipeline for moving object detection in video sequences. This project implements and benchmarks four different architectural paradigms to compare their computational efficiency:
1. **Sequential (Baseline):** Single-threaded CPU execution.
2. **OpenMP:** Multi-threaded CPU execution utilizing Just-In-Time (JIT) compilation via Numba.
3. **MPI:** Distributed-memory multi-process CPU execution via `mpi4py`.
4. **CUDA:** Hardware-accelerated mass-parallel execution on the GPU via OpenCV's transparent API.

---

## Prerequisites & Installation

### 1. Install System Dependencies
* **Windows:** Install [Microsoft MPI (MS-MPI)](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi) at the system level.
* **Linux:** Run `sudo apt install mpich libmpich-dev` or `openmpi`.
* **GPU Execution (Optional):** Requires an NVIDIA GPU with a compatible CUDA Toolkit installed.

### 2. Install Python Dependencies globally
Run the following command in your terminal/cmd to install all requirements at once:
```bash
pip install -r requirements.txt

---

## How to Run the Applications

### Sequential Version
python detection_sequential.py

### OpenMP Version
set NUMBA_NUM_THREADS=4 && python detection_openmp.py

### MPI Version
mpiexec -n 4 python detection_mpi.py

### CUDA Version
python detection_cuda.py

### Generating charts (input your data)
python generate_charts.py

### Repo structure
├── charts_results/             # Automatically generated evaluation charts (PNG)
├── ball.mp4                    # Short test video asset (121 frames)
├── traffic.mp4                 # Long Full HD test video asset (1140 frames)
├── goal.mp4                    # Low resolution / Ultra-fast test video asset (110 frames)
├── detection_sequential.py     # Sequential baseline script
├── detection_openmp.py         # OpenMP parallel implementation
├── detection_mpi.py            # MPI distributed process implementation
├── detection_cuda.py           # CUDA GPU-accelerated implementation
├── generate_charts.py          # Data analysis and 2x2 grid visualization tool
├── requirements.txt            # Python package dependencies list
└── README.md                   # Project documentation (This file)