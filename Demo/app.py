import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta, date

# Try to load TensorFlow and the LSTM model if available. Loading is optional —
# if TensorFlow is not installed or the model fails to load, the app will
# automatically fall back to simulated predictions.
MODEL = None
MODEL_INPUT_SHAPE = None
MODEL_LOADED = False
MODEL_LOAD_ERROR = None
def try_load_model():
    global MODEL, MODEL_INPUT_SHAPE, MODEL_LOADED, MODEL_LOAD_ERROR
    model_path = os.path.join('model', 'lstm.h5')
    if not os.path.exists(model_path):
        MODEL_LOADED = False
        MODEL_LOAD_ERROR = 'model file not found'
        return
    try:
        import tensorflow as tf
        # load model without compiling to avoid deserializing custom metrics
        MODEL = tf.keras.models.load_model(model_path, compile=False)
        MODEL_INPUT_SHAPE = MODEL.input_shape
        MODEL_LOADED = True
        MODEL_LOAD_ERROR = None
    except Exception as e:
        MODEL = None
        MODEL_INPUT_SHAPE = None
        MODEL_LOADED = False
        MODEL_LOAD_ERROR = str(e)

# Define the fixed location map provided by the user (moved up so preprocessing
# can reference it before it's used elsewhere).
location_map = {
    'Chơn Thành': 0,
    'Côn Sơn': 1,
    'Cần Giuộc': 2,
    'Cần Giờ': 3,
    'Củ Chi': 4,
    'Dĩ An': 5,
    'Dầu Tiếng': 6,
    'Gia Ray': 7,
    'Ho Chi Minh City': 8,
    'Hóc Môn': 9,
    'Lái Thiêu': 10,
    'Nhà Bè': 11,
    'Nhơn Trạch': 12,
    'Phuoc Tho': 13,
    'Quận Ba': 14,
    'Quận Bình Thạnh': 15,
    'Quận Bảy': 16,
    'Quận Bốn': 17,
    'Quận Chín': 18,
    'Quận Hai': 19,
    'Quận Mười': 20,
    'Quận Mười Một': 21,
    'Quận Một': 22,
    'Quận Năm': 23,
    'Quận Phú Nhuận': 24,
    'Quận Sáu': 25,
    'Quận Tân Phú': 26,
    'Thành Phố Bà Rịa': 27,
    'Thị Trấn Long Điền': 28,
    'Thị Trấn Ngải Giao': 29,
    'Thị Trấn Phú Mỹ': 30,
    'Thị Trấn Phước Bửu': 31,
    'Thủ Dầu Một': 32,
    'Thủ Đức': 33,
    'Trảng Bàng': 34,
    'Tân Phú': 35,
    'Tân Túc': 36,
    'Uyen Hung': 37,
    'Vĩnh An': 38,
    'Vũng Tàu': 39
}


st.set_page_config(page_title="AQI Prediction Demo", layout="wide")


@st.cache_data(show_spinner=False)
def load_sample_rows(path, nrows=200):
    # read a small sample to infer columns
    return pd.read_csv(path, nrows=nrows, low_memory=False)


@st.cache_data(show_spinner=False)
def load_full_data(path, datetime_col=None):
    if datetime_col:
        return pd.read_csv(path, parse_dates=[datetime_col], low_memory=False)
    else:
        return pd.read_csv(path, low_memory=False, parse_dates=True)


def infer_datetime_column(df):
    candidates = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'timestamp' in c.lower()]
    if candidates:
        return candidates[0]
    # fallback: find column that parses as datetimes
    for c in df.columns:
        try:
            pd.to_datetime(df[c].dropna().iloc[:50])
            return c
        except Exception:
            continue
    return None


def infer_station_column(df):
    names = [c for c in df.columns if any(k in c.lower() for k in ['station', 'site', 'location', 'name', 'ten', 'train'])]
    if names:
        return names[0]
    # fallback: object columns
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    if obj_cols:
        return obj_cols[0]
    return None


def infer_aqi_column(df):
    candidates = [c for c in df.columns if any(k in c.lower() for k in ['aqi', 'pm2', 'pm10', 'pm25'])]
    if candidates:
        # prefer explicit 'aqi'
        for c in candidates:
            if 'aqi' in c.lower():
                return c
        return candidates[0]
    # fallback: numeric column named value
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return num_cols[0] if num_cols else None


def aqi_category(value):
    try:
        v = float(value)
    except Exception:
        return "Không xác định"
    if v <= 50:
        return "Tốt"
    if v <= 100:
        return "Trung bình"
    if v <= 150:
        return "Kém"
    if v <= 200:
        return "Xấu"
    if v <= 300:
        return "Rất xấu"
    return "Nguy hại"


def simulate_prediction(station, selected_dt, df, aqi_col):
    # If we have actual recent values, base prediction on mean + small noise
    try:
        window_start = selected_dt - timedelta(hours=24)
        recent = df[(df['__dt'] > window_start) & (df['__dt'] <= selected_dt)]
        # Try to filter by station: prefer string match on '__station', else numeric id on '__station_id'
        if station is not None:
            if '__station' in recent.columns and station in recent['__station'].unique():
                recent = recent[recent['__station'] == station]
            elif '__station_id' in recent.columns and location_map.get(station) in recent['__station_id'].unique():
                recent = recent[recent['__station_id'] == location_map.get(station)]
        recent_s = recent[aqi_col].dropna()
        if len(recent_s) >= 3:
            base = float(recent_s.mean())
            pred = base + np.random.normal(scale=max(5, base * 0.05))
            return max(0, round(pred, 1))
    except Exception:
        pass
    # fallback: global historical average or random
    try:
        global_mean = float(df[aqi_col].dropna().mean())
        return max(0, round(global_mean + np.random.normal(scale=10), 1))
    except Exception:
        return round(np.random.uniform(10, 200), 1)


def recommend_actions(aqi_value, category):
    """Return a list of Vietnamese recommendation strings based on AQI value and category.

    Keep recommendations concise and actionable for the public and for operators.
    """
    recs = []
    try:
        aqi = float(aqi_value)
    except Exception:
        aqi = None

    if category == "Không xác định" or aqi is None:
        recs.append("Dữ liệu không đủ để đưa khuyến nghị chính xác — hãy kiểm tra cảm biến/nguồn dữ liệu.")
        recs.append("Theo dõi cập nhật từ cơ quan khí tượng/đo lường địa phương.")
        return recs

    # General recommendations by category
    if aqi <= 50:
        recs.append("Chất lượng không khí tốt — hoạt động ngoài trời bình thường.")
        recs.append("Tiếp tục theo dõi bản tin chất lượng không khí hàng ngày.")
    elif aqi <= 100:
        recs.append("Chất lượng trung bình — người nhạy cảm nên cân nhắc giảm hoạt động thể lực ngoài trời.")
        recs.append("Đóng cửa sổ khi có nhiều khói hoặc khói bụi, dùng điều hòa với lọc sạch nếu có.")
    elif aqi <= 150:
        recs.append("Chất lượng kém — hạn chế tập thể dục nặng ngoài trời.")
        recs.append("Người có bệnh hô hấp hoặc tim mạch nên ở trong nhà và tránh tiếp xúc lâu với không khí ô nhiễm.")
        recs.append("Sử dụng khẩu trang N95/FFP2 nếu phải ra ngoài trong thời gian dài.")
    elif aqi <= 200:
        recs.append("Chất lượng xấu — hạn chế mạnh hoạt động ngoài trời và các sự kiện ngoài trời.")
        recs.append("Nhóm nhạy cảm nên ở trong nhà, dùng máy lọc không khí nếu có và tham khảo ý kiến bác sĩ nếu thấy triệu chứng.")
        recs.append("Người dân nên đóng cửa nhà, tránh mở cửa sổ vào giờ ô nhiễm đỉnh điểm.")
    elif aqi <= 300:
        recs.append("Chất lượng rất xấu — tránh ra ngoài, chỉ ra ngoài khi thật cần thiết.")
        recs.append("Sử dụng khẩu trang bảo vệ hô hấp (N95/FFP2) nếu bắt buộc phải ra ngoài.")
        recs.append("Cân nhắc hoạt động trong không gian có máy lọc không khí hoặc tạo môi trường sạch trong nhà.")
    else:
        recs.append("Tình trạng nguy hại — hạn chế tối đa ra ngoài, tuân thủ hướng dẫn khẩn cấp từ cơ quan y tế địa phương.")
        recs.append("Người có triệu chứng cần liên hệ với cơ sở y tế ngay lập tức.")

    # Short recommendations for operators and longer-term actions
    if aqi is not None and aqi > 100:
        recs.append("Với mức AQI cao, các cơ quan quản lý nên rà soát nguồn phát thải và tăng cường giám sát cảm biến.")
        recs.append("Xem xét phát thông báo công cộng và khuyến cáo cho các trường học và cơ sở y tế.")

    recs.append("Theo dõi dự báo chất lượng không khí và cập nhật dự đoán khi có dữ liệu mới.")

    return recs


def main():
    st.title("Demo Dự đoán AQI — Streamlit")

    data_path = os.path.join('data', 'full_data_imputed.csv')
    has_data = os.path.exists(data_path)

    if not has_data:
        st.warning("Không tìm thấy file dữ liệu `data/full_data_imputed.csv`. Ứng dụng sẽ dùng dữ liệu giả để minh họa.")

    # sidebar inputs
    st.sidebar.header("Tùy chọn dự đoán")

    # date limits (test set as per requirement: Jul-Sep 2025)
    min_date = date(2025, 7, 1)
    max_date = date(2025, 9, 30)

    # load data (sample then full) and infer columns
    df_sample = None
    if has_data:
        try:
            df_sample = load_sample_rows(data_path, nrows=200)
        except Exception as e:
            st.error(f"Không thể đọc mẫu dữ liệu: {e}")

    datetime_col = None
    aqi_col = None

    if df_sample is not None:
        datetime_col = infer_datetime_column(df_sample)
        aqi_col = infer_aqi_column(df_sample)

    # Prepare full dataframe if possible
    df = None
    if has_data and datetime_col:
        try:
            df = load_full_data(data_path, datetime_col)
            # normalize helper column names
            df['__dt'] = pd.to_datetime(df[datetime_col])
        except Exception as e:
            st.error(f"Không thể đọc dữ liệu đầy đủ: {e}")

    # If we have the dataset, apply the same preprocessing used in training notebook
    # so we can construct model inputs compatible with the saved LSTM.
    scaler_X = None
    scaler_y = None
    model_feature_columns = None
    if df is not None:
        try:
            # Work on a copy
            df_proc = df.copy()
            # drop unnamed index column if present
            if 'Unnamed: 0' in df_proc.columns:
                df_proc = df_proc.drop(columns=['Unnamed: 0'])

            # Ensure datetime index named 'Local Time' to match notebook, but keep original '__dt'
            # If there is a 'Local Time' column, use it; otherwise create it from '__dt'
            if 'Local Time' in df_proc.columns:
                df_proc['Local Time'] = pd.to_datetime(df_proc['Local Time'])
            else:
                df_proc['Local Time'] = pd.to_datetime(df_proc['__dt'])

            df_proc = df_proc.set_index('Local Time')

            # forward/backfill weather columns used in notebook
            weather_cols = ['Clouds', 'Precipitation', 'Pressure',
                            'Relative Humidity', 'Temperature', 'UV_Index', 'Wind Speed']
            for c in weather_cols:
                if c in df_proc.columns:
                    df_proc[c] = df_proc[c].ffill().bfill()

            # create time features
            df_proc['Month'] = df_proc.index.month
            df_proc['Day'] = df_proc.index.day
            df_proc['Hour'] = df_proc.index.hour
            df_proc['Weekday'] = df_proc.index.weekday

            # drop latitude/longitude if present
            for c in ['Latitude', 'Longitude']:
                if c in df_proc.columns:
                    df_proc = df_proc.drop(columns=[c])

            # Detect station column (if any) and preserve original names
            station_col_detected = infer_station_column(df_proc)
            orig_station_series = None
            if station_col_detected and station_col_detected in df_proc.columns:
                orig_station_series = df_proc[station_col_detected].copy()

            # Encode train/station names using provided location_map if 'Train Name' exists
            if 'Train Name' in df_proc.columns:
                # map using location_map where possible, else label missing as -1
                def map_station(x):
                    return location_map.get(x, -1)
                # keep original if we don't already have it
                if orig_station_series is None:
                    orig_station_series = df_proc['Train Name'].copy()
                df_proc['Train Name'] = df_proc['Train Name'].map(map_station)

            # create helper columns for station name and station id to allow filtering later
            if orig_station_series is not None:
                # if the original station column contains numeric codes, treat them as ids
                if pd.api.types.is_numeric_dtype(orig_station_series):
                    # numeric codes likely already represent station ids
                    df_proc['__station_id'] = pd.to_numeric(orig_station_series, errors='coerce').astype('Int64')
                    # build inverse mapping from id -> name if possible
                    inv_map = {v: k for k, v in location_map.items()}
                    df_proc['__station'] = df_proc['__station_id'].map(lambda x: inv_map.get(int(x), None) if pd.notna(x) else None)
                else:
                    df_proc['__station'] = orig_station_series
                    df_proc['__station_id'] = orig_station_series.map(lambda x: location_map.get(x, -1))
            else:
                # if we only have a numeric Train Name after mapping, use it as id
                if 'Train Name' in df_proc.columns and df_proc['Train Name'].dtype in [np.int64, np.int32, np.float64]:
                    df_proc['__station_id'] = df_proc['Train Name']
                    df_proc['__station'] = None

            # target column
            target_col = 'Aqi' if 'Aqi' in df_proc.columns else ('aqi' if 'aqi' in df_proc.columns else None)

            if target_col is not None:
                # Split by date ranges to mimic training split
                train_slice = df_proc.loc['2023':'2024']
                # if not present, fall back to everything before 2025
                if train_slice.empty:
                    train_slice = df_proc[df_proc.index.year < 2025]

                # Build X_train selecting numeric features only (avoid datetime/object columns)
                X_all = train_slice.drop(columns=[target_col], errors='ignore')
                # try to select numeric columns; coerce where possible
                num_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
                if not num_cols:
                    # attempt to coerce columns to numeric where it makes sense
                    X_all = X_all.apply(pd.to_numeric, errors='coerce')
                    num_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()

                X_train = X_all[num_cols]
                y_train = train_slice[target_col] if target_col in train_slice.columns else None

                # Fit scalers same as training (StandardScaler) only on numeric features
                if y_train is not None and not X_train.empty:
                    scaler_X = StandardScaler()
                    X_train_scaled = scaler_X.fit_transform(X_train)

                    scaler_y = StandardScaler()
                    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

                    # Record feature columns order (numeric only)
                    model_feature_columns = X_train.columns.tolist()
                else:
                    scaler_X = None
                    scaler_y = None
                    # pick numeric columns from df_proc as feature candidates
                    model_feature_columns = df_proc.select_dtypes(include=[np.number]).columns.tolist()
            else:
                scaler_X = None
                scaler_y = None
                model_feature_columns = df_proc.columns.tolist()

            # store processed df for lookups later
            df = df_proc.reset_index()
            # ensure __dt is datetime (create from the Local Time column)
            if 'Local Time' in df.columns:
                df['__dt'] = pd.to_datetime(df['Local Time'])
            else:
                # fallback: try existing __dt
                if '__dt' in df.columns:
                    df['__dt'] = pd.to_datetime(df['__dt'])
        except Exception as e:
            st.sidebar.warning(f"Không thể tiền xử lý dữ liệu đầy đủ: {e}")

    # Attempt to load LSTM model (optional). This will only work if TensorFlow
    # is installed and `model/lstm.h5` exists. If not, the app will continue
    # using simulated predictions.
    try_load_model()
    if MODEL_LOADED:
        st.sidebar.success("Mô hình LSTM đã được tải và sẵn sàng sử dụng.")
    else:
        if MODEL_LOAD_ERROR == 'model file not found':
            st.sidebar.info("Không tìm thấy file `model/lstm.h5`. Ứng dụng sẽ dùng mô phỏng.")
        else:
            st.sidebar.info(f"Mô hình LSTM không được tải: {MODEL_LOAD_ERROR or 'TensorFlow chưa cài đặt'}")

    # use the global `location_map` defined at module scope

    # --- UI: only two selectboxes as requested ---
    st.markdown("### Chọn Thời gian và Trạm")

    # Build time options (prefer datetimes from dataset if available)
    time_options = []
    if df is not None and '__dt' in df.columns:
        # filter to the requested test window and round to hour
        start_dt = datetime(min_date.year, min_date.month, min_date.day, 0, 0)
        end_dt = datetime(max_date.year, max_date.month, max_date.day, 23, 0)
        mask = (df['__dt'] >= start_dt) & (df['__dt'] <= end_dt)
        times = df.loc[mask, '__dt'].dt.floor('H').drop_duplicates().sort_values()
        time_options = [t.to_pydatetime() for t in times]
    if not time_options:
        # fallback: full hourly range in Jul-Sep 2025
        cur = datetime(min_date.year, min_date.month, min_date.day, 0)
        end = datetime(max_date.year, max_date.month, max_date.day, 23)
        while cur <= end:
            time_options.append(cur)
            cur += timedelta(hours=1)

    # format for user display
    display_times = [t.strftime('%Y-%m-%d %H:00') for t in time_options]
    selected_time_str = st.selectbox("Thời gian (Ngày Tháng Năm giờ)", options=display_times, index=0)

    # station selectbox using fixed mapping
    station = st.selectbox("Trạm", options=list(location_map.keys()), index=0)

    # optional debug toggle
    debug_mode = st.sidebar.checkbox("Hiển thị debug dữ liệu (slice)", value=False)

    # Predict button (kept as a button separate from the two selectboxes)
    if st.button("Dự đoán AQI"):
        # parse selected datetime
        selected_dt = datetime.strptime(selected_time_str, '%Y-%m-%d %H:%M')

        # Attempt to predict using loaded LSTM model if available; otherwise use
        # dataset-based simulation or random simulation.
        predicted = None
        used_model = False
        if MODEL_LOADED and MODEL is not None:
            try:
                inp_shape = MODEL_INPUT_SHAPE
                # handle 3D input (batch, timesteps, features)
                if inp_shape is not None and len(inp_shape) == 3:
                    timesteps = int(inp_shape[1]) if inp_shape[1] is not None else 24
                    n_features = int(inp_shape[2]) if inp_shape[2] is not None else 1

                    # Build a feature sequence matching the model's expected features.
                    seq_arr = None
                    if df is not None and model_feature_columns is not None:
                        recent = df[df['__dt'] <= selected_dt]
                        if '__station' in df.columns and station in recent['__station'].values:
                            recent = recent[recent['__station'] == station]
                        # take the model feature columns in the same order
                        recent_X = recent.sort_values('__dt')[model_feature_columns].dropna()
                        if recent_X.shape[0] > 0:
                            if scaler_X is not None:
                                recent_X_scaled = scaler_X.transform(recent_X)
                            else:
                                recent_X_scaled = recent_X.values.astype(float)
                        else:
                            # fallback to mean row from training scaler if available
                            if scaler_X is not None:
                                mean_row = scaler_X.mean_
                                recent_X_scaled = np.tile(mean_row.reshape(1, -1), (1, 1))
                            else:
                                recent_X_scaled = np.zeros((1, len(model_feature_columns)))

                        # take last timesteps rows, pad with the mean if needed
                        if recent_X_scaled.shape[0] >= timesteps:
                            seq = recent_X_scaled[-timesteps:]
                        else:
                            pad_len = timesteps - recent_X_scaled.shape[0]
                            if recent_X_scaled.shape[0] > 0:
                                pad_row = np.mean(recent_X_scaled, axis=0, keepdims=True)
                            else:
                                pad_row = np.zeros((1, recent_X_scaled.shape[1]))
                            seq = np.vstack([np.repeat(pad_row, pad_len, axis=0), recent_X_scaled])

                        seq_arr = seq.reshape((1, timesteps, recent_X_scaled.shape[1]))
                    else:
                        # No dataset/features available; create neutral sequence
                        seq_arr = np.zeros((1, timesteps, n_features))

                    arr = seq_arr

                    pred_arr = MODEL.predict(arr)
                    # try to extract scalar prediction
                    try:
                        raw_pred = float(np.asarray(pred_arr).reshape(-1)[0])
                        # inverse-scale prediction if scaler_y available
                        if 'scaler_y' in globals() and scaler_y is not None:
                            try:
                                predicted = float(scaler_y.inverse_transform(np.array(raw_pred).reshape(-1, 1)).reshape(-1)[0])
                            except Exception:
                                predicted = raw_pred
                        else:
                            predicted = raw_pred
                        used_model = True
                    except Exception:
                        used_model = False
                else:
                    # fallback for 2D input models: build a single feature vector
                    if df is not None and model_feature_columns is not None:
                        recent = df[df['__dt'] <= selected_dt].sort_values('__dt')
                        if '__station' in df.columns and station in recent['__station'].values:
                            recent = recent[recent['__station'] == station]
                        recent_X = recent[model_feature_columns].dropna()
                        if recent_X.shape[0] > 0:
                            if scaler_X is not None:
                                recent_X_scaled = scaler_X.transform(recent_X)
                            else:
                                recent_X_scaled = recent_X.values.astype(float)
                            vec = recent_X_scaled[-1].reshape(1, -1)
                        else:
                            # fallback
                            if scaler_X is not None:
                                vec = scaler_X.mean_.reshape(1, -1)
                            else:
                                vec = np.zeros((1, len(model_feature_columns)))
                        arr = np.array(vec, dtype=float)
                    else:
                        arr = np.array([[50.0]])
                    pred_arr = MODEL.predict(arr)
                    try:
                        raw_pred = float(np.asarray(pred_arr).reshape(-1)[0])
                        if 'scaler_y' in globals() and scaler_y is not None:
                            try:
                                predicted = float(scaler_y.inverse_transform(np.array(raw_pred).reshape(-1, 1)).reshape(-1)[0])
                            except Exception:
                                predicted = raw_pred
                        else:
                            predicted = raw_pred
                        used_model = True
                    except Exception:
                        used_model = False
            except Exception as e:
                # model prediction failed; we'll fallback to simulation
                used_model = False

        if not used_model:
            if df is not None and aqi_col in df.columns:
                if '__station' not in df.columns:
                    df['__station'] = None
                predicted = simulate_prediction(station, selected_dt, df, aqi_col)
            else:
                predicted = round(np.random.uniform(10, 200), 1)

        category = aqi_category(predicted)

        # display results and 24h chart
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label="AQI dự đoán", value=str(predicted))
            st.write(f"Mức chất lượng: **{category}**")
            # show whether prediction used the real model or a simulation fallback
            src_note = "(Mô hình LSTM)" if used_model else "(Mô phỏng / fallback)"
            st.caption(f"Nguồn dự đoán: {src_note}")

            # Display short, actionable recommendations
            try:
                recs = recommend_actions(predicted, category)
                if recs:
                    st.markdown("**Khuyến nghị & biện pháp đề xuất:**")
                    for r in recs:
                        st.markdown(f"- {r}")
            except Exception as e:
                st.write("Không thể hiển thị khuyến nghị: ", e)

        with col2:
            st.subheader("Diễn biến 24 giờ gần nhất")
            chart_df = None
            if df is not None and '__dt' in df.columns and aqi_col in df.columns:
                window_start = selected_dt - timedelta(hours=24)
                slice_df = df[(df['__dt'] > window_start) & (df['__dt'] <= selected_dt)].copy()
                if not slice_df.empty:
                    # prefer filtering by station if dataset provides matching station names
                    if '__station' in slice_df.columns and station in slice_df['__station'].values:
                        slice_df = slice_df[slice_df['__station'] == station]
                    elif '__station_id' in slice_df.columns and location_map.get(station) in slice_df['__station_id'].values:
                        slice_df = slice_df[slice_df['__station_id'] == location_map.get(station)]
                    chart_df = slice_df.set_index('__dt')[[aqi_col]].rename(columns={aqi_col: 'AQI'})
                else:
                    chart_df = None
                # debug output: show raw slice and counts per hour if requested
                if debug_mode:
                    with st.expander('Debug: slice_df và thống kê giờ'):
                        try:
                            show_cols = ['__dt']
                            if '__station' in slice_df.columns:
                                show_cols.append('__station')
                            if '__station_id' in slice_df.columns:
                                show_cols.append('__station_id')
                            if aqi_col in slice_df.columns:
                                show_cols.append(aqi_col)
                            st.write('Số bản ghi trong slice_df:', len(slice_df))
                            st.dataframe(slice_df[show_cols].sort_values('__dt').reset_index(drop=True).head(200))
                            # counts per hour
                            try:
                                counts = slice_df.groupby(slice_df['__dt'].dt.floor('H')).size()
                                st.write('Số bản ghi / giờ (tối đa 50 hàng):')
                                st.dataframe(counts.reset_index(name='count').head(50))
                            except Exception as e:
                                st.write('Không thể tính counts per hour:', e)
                        except Exception as e:
                            st.write('Lỗi debug:', e)
            if chart_df is None or chart_df.empty:
                hours = [selected_dt - timedelta(hours=i) for i in range(24, -1, -1)]
                vals = (np.array(range(len(hours))) * 0.0) + predicted + np.random.normal(scale=8, size=len(hours))
                chart_df = pd.DataFrame({'datetime': hours, 'AQI': vals}).set_index('datetime')

            # Create a nicer matplotlib chart for the 24-hour AQI trend
            try:
                import matplotlib.pyplot as plt
                import matplotlib.dates as mdates

                fig, ax = plt.subplots(figsize=(9, 4))
                # ensure datetime index
                ts = chart_df.index
                vals = chart_df['AQI'].values

                ax.plot(ts, vals, marker='o', linestyle='-', color='#1f77b4', label='AQI')
                # rolling smooth
                try:
                    roll = chart_df['AQI'].rolling(window=3, min_periods=1).mean()
                    ax.plot(ts, roll.values, linestyle='--', color='#ff7f0e', label='Rolling mean (3)')
                except Exception:
                    pass

                # highlight predicted point
                ax.axvline(selected_dt, color='gray', linestyle=':', linewidth=1)
                ax.scatter([selected_dt], [predicted], color='red', zorder=5, label='Predicted')

                # grid, labels, title
                ax.grid(alpha=0.3)
                ax.set_ylabel('AQI')
                ax.set_xlabel('Thời gian')
                ax.set_title('AQI trong 24 giờ gần nhất')

                # format x axis nicely
                locator = mdates.AutoDateLocator()
                formatter = mdates.ConciseDateFormatter(locator)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)

                # y limits with margin
                ymin = float(min(vals.min(), predicted) - 5)
                ymax = float(max(vals.max(), predicted) + 5)
                ax.set_ylim(max(0, ymin), ymax)

                ax.legend(loc='upper left', fontsize='small')
                fig.tight_layout()
                st.pyplot(fig)
            except Exception:
                # fallback to the simple Streamlit chart if matplotlib unavailable
                st.line_chart(chart_df['AQI'])

    st.markdown("---")
    st.markdown("Hướng dẫn: Chọn ngày/giờ trong khoảng Jul-Sep 2025, chọn địa điểm, sau đó bấm 'Dự đoán AQI'. Ứng dụng sẽ hiển thị giá trị dự đoán (mô phỏng nếu không có mô hình) và đồ thị 24 giờ.")


if __name__ == '__main__':
    main()
