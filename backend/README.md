# Cryptocurrency Trading Bot - Backend

This is the backend component of the Cryptocurrency Trading Bot project. It handles data processing, model training, backtesting, and trading logic.

## Project Structure

- `src/data/` - Data ingestion, processing, and staging
- `src/models/` - Machine learning models (XGBoost, Neural Networks, Reinforcement Learning)
- `src/trading/` - Trading logic, backtesting, signals, and execution
- `src/api/` - API endpoints for frontend communication
- `src/shared/` - Shared utilities, configuration, and logging
- `src/pipeline/` - Orchestration and workflow management
- `tests/` - Test files

## Setup

1. Create a virtual environment:
```
python -m venv venv
```

2. Activate the virtual environment:
```
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

## Running the Application

To run the main pipeline:
```
python -m src.pipeline.main
```

## Development

The codebase follows a modular architecture with clear separation of concerns:
- Data processing is separated from model training
- Trading logic is independent of data processing
- Configuration is centralized in the shared directory
- Utilities are organized by functionality

## Future Enhancements

- Neural Network models
- Reinforcement Learning models
- Real-time data ingestion
- API endpoints for frontend communication 