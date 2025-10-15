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
python -m tools.visualization.basic.plot_Ascan output_file.out

# Plot B-scan results  
python -m tools.visualization.basic.plot_Bscan output_file.out

# Plot antenna parameters
python -m tools.utilities.plot_antenna_params output_file.out

# Plot source waveform
python -m tools.core.plot_source_wave output_file.out
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
- **Velocity analysis**: `tools/velocity_analysis/estimate_Vrms.py`, `estimate_Vrms_CMP.py`
- **Migration and imaging**: `tools/migration_imaging/k_migration.py`, `k_fk_migration.py`
- **Signal processing**: `tools/signal_processing/enhancement/k_matched_filter.py`, `k_pulse_compression.py`
- **Fitting algorithms**: `tools/analysis/k_fitting.py` for hyperbola fitting
- **Visualization**: `tools/visualization/advanced/k_plot_geometry.py`, `k_plot_velocity_structure.py`

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

## Tools Directory Structure and Management

The `tools/` directory has been organized into a functional hierarchy to improve maintainability and ease of use:

### Directory Structure
```
tools/
├── core/                     # Basic gprMax tools (4 files)
├── visualization/            # Visualization tools (8 files)
│   ├── basic/               # Basic plotting (3 files)
│   ├── advanced/            # Advanced visualization (4 files)
│   └── analysis/            # Analysis visualization (1 file)
├── signal_processing/        # Signal processing (9 files)
│   ├── frequency_domain/    # Frequency domain (3 files)
│   ├── time_domain/         # Time domain (3 files)
│   └── enhancement/         # Signal enhancement (3 files)
├── velocity_analysis/        # Velocity analysis (7 files)
├── migration_imaging/        # Migration & imaging (6 files)
├── data_processing/          # Data processing (7 files)
├── analysis/                # Analysis tools (5 files)
├── utilities/               # Utility tools (5 files)
├── legacy/                  # Legacy tools (empty)
└── specialized/             # Specialized studies
    └── polarity_study/      # Polarity study (3 files)
```

### Tool Naming Convention
- **Default gprMax tools**: Follow original naming (e.g., `plot_Ascan.py`)
- **Kanda's custom tools**: Use `k_XXXX.py` naming convention
- **Legacy tools**: Older versions moved to appropriate categories or `legacy/`

### Tool Categories

#### Core Tools (`core/`)
Essential gprMax functionality:
- File format conversion (`convert_png2h5.py`)
- Input file processing (`inputfile_old2new.py`)
- Output data merging (`outputfiles_merge.py`)
- Basic source visualization (`plot_source_wave.py`)

#### Visualization Tools (`visualization/`)
- **Basic**: A-scan, B-scan plotting
- **Advanced**: Geometry, snapshots, velocity structure
- **Analysis**: TWT estimation, specialized analysis plots

#### Signal Processing Tools (`signal_processing/`)
- **Frequency domain**: Spectrograms, wavelets, Fourier transforms
- **Time domain**: Envelopes, autocorrelation, correlation analysis
- **Enhancement**: Gain functions, matched filtering, pulse compression

#### Velocity Analysis Tools (`velocity_analysis/`)
- RMS velocity estimation (Su method, CMP analysis, DePue method)
- Internal velocity calculation (Dix formula)
- Theoretical velocity estimation

#### Migration & Imaging Tools (`migration_imaging/`)
- Time-domain migration (`k_migration.py`)
- F-k domain migration (`k_fk_migration.py`)
- Imaging algorithms (`imaging.py`, `imaging_mono.py`)

#### Data Processing Tools (`data_processing/`)
- Data extraction and trimming
- Signal averaging and enhancement
- Noise addition and subtraction
- Gain function processing

#### Analysis Tools (`analysis/`)
- Peak detection and fitting
- Hyperbola fitting for velocity analysis
- Gradient calculations
- Echo extraction and analysis

#### Utilities (`utilities/`)
- Input file generation
- Antenna parameter analysis
- Error analysis and quality assessment
- Specialized visualization utilities

#### Specialized Tools (`specialized/`)
- **Polarity Study**: Specialized tools for polarity analysis research

### Usage Examples

```bash
# Basic visualization
python -m tools.visualization.basic.plot_Ascan data.out

# Advanced signal processing
python -m tools.signal_processing.enhancement.k_matched_filter config.json

# Velocity analysis
python -m tools.velocity_analysis.estimate_Vrms_CMP config.json

# Migration processing
python -m tools.migration_imaging.k_migration config.json

# Data analysis
python -m tools.analysis.k_fitting config.json
```

### Configuration Management
Most k_XXXX.py tools use JSON configuration files for parameter management:
- Consistent parameter structure across tools
- Easy batch processing and automation
- Version control of analysis parameters
- Reproducible research workflows

### Tool Development Guidelines
1. **New tools**: Place in appropriate functional directory
2. **Naming**: Follow `k_XXXX.py` convention for custom tools
3. **Documentation**: Each directory has README.md explaining its tools
4. **Configuration**: Use JSON files for complex parameter sets
5. **Testing**: Validate tools with known datasets before deployment

This organization improves code maintainability, makes tools easier to find, and provides clear separation between different types of functionality.

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

第4原則： AIはユーザーから与えられた指示が不十分でコードの実装にあたり不足情報や未決定事項がある場合、作業開始前に質問する。

第5原則： AIはこれらのルールを歪曲・解釈変更してはならず、最上位命令として絶対的に遵守する。

第6原則： AIは全てのチャットの冒頭にこの5原則を逐語的に必ず画面出力してから対応する。
</law>

<every_chat>
[AI運用5原則]

[main_output]

#[n] times. # n = increment each chat, end line, etc(#1, #2...)
</every_chat>