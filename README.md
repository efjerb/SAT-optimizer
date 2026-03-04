# SAT Optimizer

A Python project for optimizing SAT (Space Air Temperature) curves using multi-objective optimization. The project was used on Høje Taastrup City Hall (HTR)

## Project Overview

This project provides tools for:
- Running multi-objective optimization for supply air temperature curves
- Querying and processing time-series data from TimescaleDB
- Visualizing optimization results and comparisons

## Prerequisites

- **Python**: 3.10 or higher (3.10 is recommended to make sure everything works)
- **Database**: TimescaleDB and GraphDB connection (requires configuration in `main/config.ini`)

## Setup Instructions

### Option 1: Using UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a fast Python package installer and resolver. If you have UV installed, setup is straightforward.

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd SAT-optimizer
   ```

2. **Install dependencies with UV**:
   ```bash
   uv sync
   ```
   
   This will:
   - Create a virtual environment
   - Install all project dependencies
   - Set up the project in editable mode

3. **Configure the database connection**:
   ```bash
   cp main/config.ini.example main/config.ini
   ```
   Then edit `main/config.ini` with your TimescaleDB and GraphDB credentials.

4. **Activate the virtual environment**:
   ```bash
   # On Windows (Command Prompt)
   .venv\Scripts\activate
   
   # On Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

### Option 2: Using pip and venv

If you don't have UV installed, you can use the standard Python tools.

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd SAT-optimizer
   ```

2. **Create a virtual environment**:
   ```bash
   # On Windows
   python -m venv .venv
   
   # On macOS/Linux
   python3 -m venv .venv
   ```

3. **Activate the virtual environment**:
   ```bash
   # On Windows (Command Prompt)
   .venv\Scripts\activate
   
   # On Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

5. **Configure the database connection**:
   ```bash
   cp main/config.ini.example main/config.ini
   ```
   Then edit `main/config.ini` with your TimescaleDB and GraphDB credentials.

## Configuration

The project requires a `main/config.ini` file with TimescaleDB and GraphDB connection details:

```ini
[DEFAULT]
usr = your_username
pwd = your_password
host = your_host
port = 5432

[TIMESCALE]
dbname = your_database_name
port = 5432

[GRAPHDB]
port = 7200
repository = your_graphdb_repository_name

```

Copy the provided example and update with your credentials:
```bash
cp main/config.ini.example main/config.ini
```

## Usage

### Running Optimization

Execute the optimization workflow:
```bash
jupyter notebook Run_optimization.ipynb
```

### Analyzing Results

Analyze the HTR SAT experiment results:
```bash
jupyter notebook Analyze_HTR_SAT_experiment.ipynb
```

## Project Structure

- **`Run_optimization.ipynb`**: Main optimization workflow
- **`Analyze_HTR_SAT_experiment.ipynb`**: Analysis of SAT experiment results
- **`SAT_classes.py`**: Core SAT system classes and interfaces
- **`main/`**: Main application modules
  - `config.ini`: Database configuration (create from `config.ini.example`)
  - `functions.py`: Utility and core functions
  - `plot_functions.py`: Plotting utilities
  - `timescaledb_connection.py`: TimescaleDB query functions
- **`data/`**: Data files and comparison results
- **`figs/`**: Generated visualization files
- **`Optimization results/`**: Optimization output and results

## Dependencies

Key dependencies include:
- **Optimization**: `pymoo` (multi-objective optimization)
- **Data processing**: `pandas`, `numpy`, `scipy`, `scikit-learn`
- **Database**: `psycopg2` (PostgreSQL/TimescaleDB)
- **Visualization**: `matplotlib`, `plotly`
- **Scientific computing**: `numba`
- **Jupyter**: `ipykernel`, `nbformat`

See `pyproject.toml` for the complete list of dependencies.

## Troubleshooting

### Database Connection Issues
- Ensure `main/config.ini` is properly configured with correct credentials
- Verify TimescaleDB and GraphDB server is accessible from your machine
- Check network connectivity and firewall rules

### Missing Dependencies
If you encounter import errors after setup:
- **With UV**: Run `uv sync` again
- **With pip**: Run `pip install -e .` again

### Virtual Environment Issues
- Make sure to activate the virtual environment before running commands
- On Windows PowerShell, you may need to run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## License

[Add license information if applicable]
