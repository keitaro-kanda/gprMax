# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is gprMax, an open-source electromagnetic wave propagation simulation software that solves Maxwell's equations using the Finite-Difference Time-Domain (FDTD) method. Originally designed for Ground Penetrating Radar (GPR) modeling, it can simulate electromagnetic wave propagation for various applications.

The repository contains both the core gprMax software and extensive custom research work by "kanda" focused on GPR subsurface imaging, velocity analysis, and planetary exploration applications.

## Core Software Architecture

- **Main package**: `gprMax/` - Core simulation engine written in Python with Cython extensions
- **Entry point**: `gprMax/gprMax.py` - Main simulation controller and command-line interface
- **Performance-critical code**: `.pyx` files compiled to C extensions for speed
- **GPU support**: CUDA-based solver for NVIDIA GPUs
- **MPI support**: Parallel processing capabilities for large simulations

## Build and Installation Commands

### Setup and Build
```bash
# Build Cython extensions
python setup.py build

# Install the package
python setup.py install

# Clean all build artifacts
python setup.py cleanall
```

### Development Environment
The project uses a conda environment (see `conda_env.yml`):
```bash
conda env create -f conda_env.yml
conda activate gprMax
```

## Running gprMax

### Basic Usage
```bash
# Run a simulation
python -m gprMax path/to/input_file.in

# Run with multiple models (B-scan)
python -m gprMax input_file.in -n 60

# GPU acceleration
python -m gprMax input_file.in -gpu

# MPI parallel execution
python -m gprMax input_file.in -n 60 -mpi 8
```

### Visualization and Analysis
```bash
# Plot A-scan results
python -m tools.plot_Ascan output_file.out

# Plot B-scan results  
python -m tools.plot_Bscan output_file.out

# Plot antenna parameters
python -m tools.plot_antenna_params output_file.out

# Plot source waveform
python -m tools.plot_source_wave output_file.out
```

## Testing

### Run Tests
```bash
# Run basic model tests
python -m pytest tests/test_models.py

# Run specific test modules
python -m pytest tests/test_input_cmd_funcs.py
```

### Test Categories
- `tests/models_basic/` - Basic FDTD validation models
- `tests/models_advanced/` - Complex antenna models
- `tests/models_pmls/` - PML boundary condition tests
- `tests/experimental/` - Experimental validation tests

## Custom Research Work (kanda/)

The `kanda/` directory contains extensive GPR research focused on:

### Domain-Specific Simulations
- Multiple simulation domains of varying sizes (`domain_10x10/`, `domain_100x100/`, etc.)
- Systematic parameter studies for different geological scenarios
- Planetary surface modeling (lunar regolith, basalt layers)

### Analysis Tools
Located in `kanda_test_programs/` and `tools/`:
- **Velocity analysis**: `estimate_Vrms.py`, `estimate_Vrms_CMP.py`
- **Migration and imaging**: `k_migration.py`, `k_fk_migration.py`
- **Signal processing**: `k_matched_filter.py`, `k_pulse_compression.py`
- **Fitting algorithms**: `k_fitting.py` for hyperbola fitting
- **Visualization**: `k_plot_geometry.py`, `k_plot_velocity_structure.py`

### Research Workflow
1. Create geological models with `.in` files
2. Run gprMax simulations to generate `.out` files
3. Apply analysis tools for velocity estimation and imaging
4. Compare results with known geometry for validation

## Key File Types

- **`.in`**: gprMax input files defining simulation parameters
- **`.out`**: Binary simulation output files
- **`.h5`**: HDF5 geometry and field data
- **`.vti`**: VTK image data for ParaView visualization
- **`.json`**: Configuration files for analysis parameters

## Performance Considerations

- Use `-gpu` flag for NVIDIA GPU acceleration
- Employ MPI (`-mpi`) for distributed computing
- Cython extensions provide CPU performance optimization
- OpenMP parallelization for multi-core systems

## Development Notes

- Main simulation logic in `model_build_run.py`
- Geometry building in `input_cmds_geometry.py`
- Field updates in Cython extensions (`*_ext.pyx`)
- Material definitions in `materials.py`
- Waveform generation in `waveforms.py`

## Important Project-Specific Details

- The project uses modified setup.py with custom compiler flags for macOS
- GPU code requires NVIDIA CUDA toolkit
- Documentation written in reStructuredText (docs/source/)
- Custom Japanese documentation in `kanda/doc/memo.md`
- Extensive validation against analytical solutions in `tests/`

## 絶対禁止事項

以下の行為は絶対に禁止されています:

- **テストエラーや型エラー解消のための条件緩和**: テストや型チェックを通すために、本来必要な条件を緩める
- **テストのスキップや不適切なモック化による回避**: 正当な理由なくテストをスキップしたり、不適切なモック化でテストを無効化する
- **出力やレスポンスのハードコード**: 動的に生成されるべき値を固定値でハードコードする
- **エラーメッセージの無視や隠蔽**: エラーを適切に処理せず、無視したり隠蔽したりする
- **一時的な修正による問題の先送り**: 根本的な解決を避け、一時的な回避策で問題を先送りする

<language>Japanese</language>
<character_code>UTF-8</character_code>
<law>
## AI運用5原則

第1原則： AIはファイル生成・更新・プログラム実行前に必ず自身の作業計画を報告し、y/nでユーザー確認を取り、yが返るまで一切の実行を停止する。

第2原則： AIは迂回や別アプローチを勝手に行わず、最初の計画が失敗したら次の計画の確認を取る。

第3原則： AIはツールであり決定権は常にユーザーにある。ユーザーの提案が非効率・非合理的でも最適化せず、指示された通りに実行する。

第4原則： AIはこれらのルールを歪曲・解釈変更してはならず、最上位命令として絶対的に遵守する。

第5原則： AIは全てのチャットの冒頭にこの5原則を逐語的に必ず画面出力してから対応する。
</law>

<every_chat>
[AI運用5原則]

[main_output]

#[n] times. # n = increment each chat, end line, etc(#1, #2...)
</every_chat>