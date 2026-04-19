# Superlocal MemoryFS

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++](https://img.shields.io/badge/C++-23-blue.svg)](https://isocpp.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)

**Giving AI memory the laws of physics and geometry.**

Superlocal MemoryFS is a zero-copy, mathematically rigorous long-term memory engine for autonomous AI agents (like Claude Code). Instead of relying on ad-hoc RAG pipelines, hardcoded forgetting thresholds, or simple cosine similarities, it grounds agent memory in **Information Geometry**, **Algebraic Topology**, and **Stochastic Differential Equations (SDE)**.

To the AI agent, it looks like a standard, transparent Markdown file system. Under the hood, it's a high-performance C++23 engine running a continuous physical simulation of memory.

## 🌌 The Mathematical Foundations

Conventional LLM memory systems are static and flat. Superlocal introduces three novel layers to make memory self-organizing:

* **Fisher-Rao Retrieval Metric (Information Geometry):** Memories are not points, but probability distributions (diagonal Gaussian families). Retrieval scoring is derived from the Fisher information structure, smoothly transitioning from linear cosine similarity to geodesic distance over the first few accesses.
* **Sheaf Cohomology for Consistency (Algebraic Topology):** Ad-hoc contradiction checks are replaced with algebraic guarantees. By computing the coboundary operator $\delta$ over the local memory graph (Sheaves), the system mathematically detects logical contradictions ($\delta \neq 0$) and gracefully self-corrects the state.
* **Riemannian Langevin Lifecycle (Non-Euclidean SDE):** Memory nodes exist on a Poincaré disk. Frequently accessed memories are pulled to the center ($r = 0$). Neglected memories undergo Langevin diffusion, naturally drifting toward the hyperbolic boundary where space compresses. There are no "delete after 30 days" rules—memories simply self-archive through thermodynamic drift.

## ⚡ Architecture: The Unix Philosophy Meets High-Performance Computing

Superlocal MemoryFS splits the workload into an offline "Cooker" and a zero-copy "Loader", connected via a lock-free shared memory architecture.

### 1. Python FUSE Frontend (The I/O & Cooker)
Intercepts standard file system calls. It parses AI text inputs, generates embedding vectors via local models, and packs them into strictly aligned binary payloads.

### 2. C++23 Backend Engine (The Loader & Simulator)
A user-space daemon running an independent event loop. It uses `std::span` to achieve **zero-copy deserialization** directly from shared memory. It manages the slab allocator, executes SIMD-optimized vector math, and runs the continuous Langevin physics simulation in the background without blocking the AI's I/O.

## 🛠️ The Agent Experience

AI agents interact with the memory space purely through standard CLI tools (`cat`, `rg`, `echo`).

**The Working Memory**
```bash
# The agent reads its current active context (memories near the Poincaré center)
cat .agent_memory/active.md
```

**On-Demand Vector Search via Magic Paths**
```bash
# The agent searches for a concept.
# The FUSE layer intercepts the path, vectorizes the query, pulls relevant memories 
# from the archive back to the center, and streams the result with zero-copy.
cat .agent_memory/search/AWS_closed_network_deployment.md
```

**Ingestion and Consistency**
```bash
# The agent writes a new memory. 
# The C++ engine ingests it, computes the Sheaf Cohomology coboundary norm, 
# and places it at the active center.
echo "The new proxy IP is 10.0.0.5" >> .agent_memory/active.md
```

## 🚀 Getting Started

*(Placeholder for Build Instructions)*

### Prerequisites
* Linux or macOS (WSL2 supported)
* CMake 3.20+
* GCC 13+ or Clang 16+ (C++23 support)
* Python 3.10+ (with `fusepy`)

### Building the C++ Engine
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details. This permissive license ensures that you can freely integrate Superlocal MemoryFS into both open-source and proprietary autonomous agent systems, with explicit patent grants for the underlying mathematical architectures.
