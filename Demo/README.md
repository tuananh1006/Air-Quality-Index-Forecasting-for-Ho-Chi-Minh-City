# AQI Prediction Demo (Streamlit)

This is a small Streamlit demo app that illustrates how to predict AQI (Air Quality Index) for a chosen location and datetime.

Files:
- `app.py`: main Streamlit application. Run with `streamlit run app.py`.
- `data/full_data_imputed.csv`: (optional) dataset used for realistic examples. If missing, the app simulates data.
- `model/lstm.h5`: (optional) example model file. The demo currently simulates predictions; you can extend `app.py` to load and use this model.
- `requirements.txt`: Python dependencies.

Usage:

1. Install dependencies (recommended inside a virtualenv):

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run app.py
```

Notes:
- The UI restricts date selection to July–September 2025 as requested. If your CSV contains actual data, the app will try to infer the datetime, station, and AQI/value columns automatically. You can override those in the sidebar.
- Prediction is simulated by default. If you want to use `model/lstm.h5`, install TensorFlow and modify `app.py` to load and prepare features according to your model input.
