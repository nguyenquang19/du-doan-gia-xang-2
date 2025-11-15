import pandas as pd
import numpy as np
import streamlit as st
import os
import matplotlib.pyplot as plt

# --- Imports cho các mô hình Scikit-learn ---
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# --- Imports cho các mô hình bên ngoài (CẦN PHẢI CÀI ĐẶT BẰNG PIP) ---
try:
    import xgboost as xgb
    XGBRegressor = xgb.XGBRegressor
except Exception:
    XGBRegressor = None

try:
    import lightgbm as lgb
    LGBMRegressor = lgb.LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    import catboost as cat
    CatBoostRegressor = cat.CatBoostRegressor
except Exception:
    CatBoostRegressor = None

# -----------------------------------------------------------------------------------
# THAM SỐ CẤU HÌNH VÀ TÊN FILE
# -----------------------------------------------------------------------------------
RAW_DATA_FILE = "Data_tho_chua_xu_ly.csv"
TARGET_COL = 'RON 95-III(VND)'
TEST_SIZE = 150
LAG_W = [1, 7]
VOL_W = 7
EVENT_LAG = [3, 7]

EVENT_MAP = {
    'Cung (OPEC & Sản lượng)': 'event_Cung (OPEC & Sản lượng)',
    'Cung (Tồn kho Mỹ)': 'event_Cung (Tồn kho Mỹ)',
    'Cầu (Kinh tế vĩ mô)': 'event_Cầu (Kinh tế vĩ mô)',
    'Sự cố & Gián đoạn': 'event_Sự cố & Gián đoạn',
    'Địa chính trị & Xung đột': 'event_Địa chính trị & Xung đột',
    'Đồng USD & Tài chính': 'event_Đồng USD & Tài chính'
}

# -----------------------------------------------------------------------------------
# A. HÀM FEATURE ENGINEERING VÀ SCALING (AN TOÀN VỚI TÊN CỘT)
# -----------------------------------------------------------------------------------
def create_features(df_raw, scaler=None, fit_scaler=False):
    """Thực hiện toàn bộ quá trình Feature Engineering và Scaling/Transforming.
    Trả về:
      - nếu fit_scaler=True: (X_scaled_df, y_series, scaler)
      - elif scaler provided: (X_scaled_df, y_series)
      - else: (X_features_df, y_series)
    """
    df = df_raw.copy().reset_index(drop=True)
    df.columns = df.columns.astype(str)

    # đảm bảo có cột date
    if 'date' not in df.columns:
        raise ValueError("Column 'date' không tồn tại trong dữ liệu đầu vào.")

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Fill forward/backward các cột giá quan trọng nếu tồn tại
    cols_to_fill = [c for c in ['Gia_Brent(USD)', 'Gia_WTI(USD)', 'USD/VND', 'Bien_loi_nhuan'] if c in df.columns]
    if cols_to_fill:
        df[cols_to_fill] = df[cols_to_fill].ffill().bfill()

    # Drop các cột không cần thiết nếu tồn tại
    for c in ['E5 RON 92-II(VND)', 'Bien_loi_nhuan']:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Các cột giá cơ bản — kiểm tra tồn tại
    price_cols = [c for c in ['Gia_Brent(USD)', 'Gia_WTI(USD)', 'USD/VND'] if c in df.columns]
    if not price_cols:
        raise ValueError("Không tìm thấy cột giá nào trong dữ liệu (Gia_Brent/WTI/USD/VND).")

    # Tạo lag, pct change, volatility
    for col in price_cols:
        base = col.split("(")[0].rstrip()
        # lags
        for lag in LAG_W:
            df[f'{base}_lag{lag}'] = df[col].shift(lag)
        # percent change
        df[f'{base}_pct'] = df[col].pct_change()
        # rolling volatility
        df[f'{base}_vol{VOL_W}'] = df[col].rolling(window=VOL_W, min_periods=1).std()

    # Sự kiện và sentiment: chuẩn hóa cột có thể thiếu
    if 'loai_su_kien' not in df.columns:
        df['loai_su_kien'] = np.nan
    if 'tang_giam' not in df.columns:
        df['tang_giam'] = np.nan
    if 'ten_su_kien' not in df.columns:
        df['ten_su_kien'] = np.nan

    df['loai_su_kien'] = df['loai_su_kien'].fillna('No_Event')
    df['tang_giam'] = df['tang_giam'].fillna('None')

    # One-hot event categories; đảm bảo có đủ các cột theo EVENT_MAP
    event_dummies = pd.get_dummies(df['loai_su_kien'].astype(str)).astype(int)
    # rename keys present to our standardized names
    rename_map = {k: v for k, v in EVENT_MAP.items() if k in event_dummies.columns}
    if rename_map:
        event_dummies = event_dummies.rename(columns=rename_map)
    # add any missing event columns with zeros
    for std_col in EVENT_MAP.values():
        if std_col not in event_dummies.columns:
            event_dummies[std_col] = 0

    # drop No_Event column if present
    if 'No_Event' in event_dummies.columns:
        event_dummies = event_dummies.drop(columns=['No_Event'])

    df['event_impact'] = (df['loai_su_kien'] != 'No_Event').astype(int)

    sentiment_map = {'Giảm': -1, 'Tăng': 1, 'None': 0}
    df['sentiment_score'] = df['tang_giam'].map(sentiment_map).fillna(0).astype(int)
    df['event_sentiment_7'] = df['sentiment_score'].rolling(window=VOL_W, min_periods=1).sum()
    # Now event lag features (rolling sum of event_impact shifted by 1 day)
    for lag in EVENT_LAG:
        df[f'event_lag_{lag}'] = df['event_impact'].shift(1).rolling(window=lag, min_periods=1).sum().fillna(0).astype(int)

    # Combine features: drop textual columns
    drop_cols = [c for c in ['loai_su_kien', 'tang_giam', 'ten_su_kien'] if c in df.columns]
    df_features = pd.concat([df.drop(columns=drop_cols + [TARGET_COL] if TARGET_COL in df.columns else drop_cols, errors='ignore'),
                             event_dummies.reset_index(drop=True)], axis=1)

    # Drop rows with NaN in essential feature columns (after creating lags)
    df_features = df_features.dropna().reset_index(drop=True)

    # Target series (aligned with df_features index)
    if TARGET_COL in df.columns:
        y_raw = df.loc[df_features.index, TARGET_COL].reset_index(drop=True)
    else:
        # nếu không có target trong df (ví dụ khi thêm hàng input chưa có giá), tạo series NaN
        y_raw = pd.Series([np.nan] * len(df_features), name=TARGET_COL)

    # Final X (drop date column from features but keep as index)
    if 'date' in df_features.columns:
        X_features = df_features.drop(columns=['date'])
    else:
        X_features = df_features.copy()

    # Standard scaling when yêu cầu
    if fit_scaler:
        scaler_obj = StandardScaler()
        X_scaled = scaler_obj.fit_transform(X_features)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_features.columns)
        return X_scaled_df, y_raw, scaler_obj

    if scaler is not None:
        X_scaled = scaler.transform(X_features)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_features.columns)
        return X_scaled_df, y_raw

    return X_features, y_raw

# -----------------------------------------------------------------------------------
# B. HÀM TẢI VÀ HUẤN LUYỆN NHIỀU MÔ HÌNH
# -----------------------------------------------------------------------------------
@st.cache_resource
def load_and_train_model():
    """Tải dữ liệu, chuẩn bị, và huấn luyện nhiều mô hình."""
    if not os.path.exists(RAW_DATA_FILE):
        st.error(f"File dataset '{RAW_DATA_FILE}' không tìm thấy.")
        return None, None, None, None, None

    df_raw = pd.read_csv(RAW_DATA_FILE)
    df_raw.columns = df_raw.columns.astype(str)

    # Fit scaler và tạo feature matrix
    try:
        X_scaled, y_raw, scaler = create_features(df_raw, fit_scaler=True)
    except Exception as e:
        st.error(f"Lỗi khi tạo feature: {e}")
        return None, None, None, None, None

    # train/test split (index based on sorted date prior)
    if len(X_scaled) <= TEST_SIZE + 10:
        st.warning("Dữ liệu quá ít so với TEST_SIZE — giảm TEST_SIZE hoặc bổ sung dữ liệu.")
    X_train = X_scaled.iloc[:-TEST_SIZE]
    X_test = X_scaled.iloc[-TEST_SIZE:]
    y_train = y_raw.iloc[:-TEST_SIZE]
    y_test = y_raw.iloc[-TEST_SIZE:]

    models = {
        "Random Forest Regressor": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    }

    if XGBRegressor is not None:
        models["XGBoost Regressor"] = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
    if LGBMRegressor is not None:
        models["LightGBM Regressor"] = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    if CatBoostRegressor is not None:
        models["CatBoost Regressor"] = CatBoostRegressor(iterations=100, random_state=42, verbose=0)

    model_results = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train.values)
            y_pred_test = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            model_results[name] = {'model': model, 'rmse': rmse}
        except Exception as e:
            # Không dừng toàn bộ pipeline nếu 1 model lỗi
            st.warning(f"Không thể huấn luyện mô hình {name}. Lỗi: {e}")

    if not model_results:
        st.error("Không có mô hình nào được huấn luyện thành công.")
        return None, None, None, None, None

    best_model_name = min(model_results, key=lambda k: model_results[k]['rmse'])
    feature_names = X_scaled.columns.tolist()

    return model_results, best_model_name, feature_names, scaler, df_raw

# -----------------------------------------------------------------------------------
# C. HÀM DỰ ĐOÁN VỚI INPUT THÔ (Single-step)
# -----------------------------------------------------------------------------------
def predict_raw_input(raw_input_dict, df_raw_full, feature_names, scaler, selected_model):
    """Dự đoán từ input thô của người dùng (1 bước)."""
    # tạo bản sao dữ liệu lịch sử và thêm 1 hàng input (không làm thay đổi df_raw_full gốc)
    df_history = df_raw_full.copy().reset_index(drop=True)
    new_row = {
        'date': pd.to_datetime(raw_input_dict.get('date')),
        'Gia_Brent(USD)': raw_input_dict.get('Gia_Brent(USD)', np.nan),
        'Gia_WTI(USD)': raw_input_dict.get('Gia_WTI(USD)', np.nan),
        'USD/VND': raw_input_dict.get('USD/VND', np.nan),
        'loai_su_kien': raw_input_dict.get('loai_su_kien', np.nan),
        'ten_su_kien': raw_input_dict.get('ten_su_kien', np.nan),
        'tang_giam': raw_input_dict.get('tang_giam', np.nan),
        # target and other numeric fields can be NaN
        'E5 RON 92-II(VND)': np.nan,
        'RON 95-III(VND)': np.nan,
        'Bien_loi_nhuan': np.nan
    }
    df_history = pd.concat([df_history, pd.DataFrame([new_row])], ignore_index=True)

    # tạo features (sử dụng scaler đã fit)
    X_full, _ = create_features(df_history, scaler=scaler, fit_scaler=False)
    # Lấy hàng dự đoán cuối cùng
    X_predict = X_full.iloc[[-1]]
    # đảm bảo cùng thứ tự feature_names
    X_predict = X_predict.reindex(columns=feature_names, fill_value=0)
    raw_prediction = selected_model.predict(X_predict)[0]
    return raw_prediction, X_predict

# -----------------------------------------------------------------------------------
# D. HÀM DỰ BÁO ĐỆ QUY VÀ BOOTSTRAP CI (Multi-step Forecast)
# -----------------------------------------------------------------------------------
def recursive_forecast(df_raw_full, feature_names, scaler, selected_model, forecast_steps=30, n_bootstraps=30, st_container=None):
    """Dự báo đệ quy và ước lượng CI bằng bootstrap.
       Nếu truyền st_container (ví dụ st) sẽ hiện progress bar.
    """
    from collections import OrderedDict

    df_history = df_raw_full.copy().reset_index(drop=True)
    # đảm bảo có cột date
    df_history['date'] = pd.to_datetime(df_history['date'])
    df_history = df_history.sort_values('date').reset_index(drop=True)

    all_predictions = OrderedDict()

    progress = None
    if st_container is not None:
        progress = st_container.progress(0)

    for step in range(1, forecast_steps + 1):
        # next date
        last_date = df_history['date'].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)

        # Prepare next input by carrying forward last known numeric macro values and event info.
        next_input = {
            'date': next_date,
            'Gia_Brent(USD)': df_history['Gia_Brent(USD)'].iloc[-1] if 'Gia_Brent(USD)' in df_history.columns else np.nan,
            'Gia_WTI(USD)': df_history['Gia_WTI(USD)'].iloc[-1] if 'Gia_WTI(USD)' in df_history.columns else np.nan,
            'USD/VND': df_history['USD/VND'].iloc[-1] if 'USD/VND' in df_history.columns else np.nan,
            'loai_su_kien': df_history['loai_su_kien'].iloc[-1] if 'loai_su_kien' in df_history.columns else np.nan,
            'ten_su_kien': np.nan,
            'tang_giam': df_history['tang_giam'].iloc[-1] if 'tang_giam' in df_history.columns else np.nan,
            'E5 RON 92-II(VND)': np.nan,
            'RON 95-III(VND)': np.nan,
            'Bien_loi_nhuan': np.nan
        }

        # append new input (without target) so feature builder can compute lags
        df_history = pd.concat([df_history, pd.DataFrame([next_input])], ignore_index=True)

        # bootstrap predictions
        bootstrap_preds = []
        for i in range(n_bootstraps):
            X_full, _ = create_features(df_history, scaler=scaler, fit_scaler=False)
            X_predict = X_full.iloc[[-1]].reindex(columns=feature_names, fill_value=0)
            pred = selected_model.predict(X_predict)[0]
            bootstrap_preds.append(pred)

        mean_pred = float(np.mean(bootstrap_preds))
        lower_ci = float(np.percentile(bootstrap_preds, 2.5))
        upper_ci = float(np.percentile(bootstrap_preds, 97.5))

        all_predictions[next_date] = {'Giá Dự báo': mean_pred, 'CI 95% Min': lower_ci, 'CI 95% Max': upper_ci}

        # fill predicted mean vào lịch sử (cho bước tiếp theo dùng làm lag)
        df_history.loc[df_history.index[-1], 'RON 95-III(VND)'] = mean_pred

        if progress is not None:
            progress.progress(int(step / forecast_steps * 100))

    if progress is not None:
        progress.empty()

    df_forecast_results = pd.DataFrame.from_dict(all_predictions, orient='index')
    df_forecast_results.index.name = 'Ngày'
    return df_forecast_results

# -----------------------------------------------------------------------------------
# E. HÀM PHÂN TÍCH XU HƯỚNG LỊCH SỬ (Historical Trends)
# -----------------------------------------------------------------------------------
def plot_historical_trends(df_raw, days=90):
    df = df_raw.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')

    cols_to_fill = [c for c in ['Gia_Brent(USD)', 'Gia_WTI(USD)', 'RON 95-III(VND)'] if c in df.columns]
    df[cols_to_fill] = df[cols_to_fill].ffill().bfill()

    price_cols = cols_to_fill
    df_trends = df[price_cols].tail(days)
    df_pct_change = df_trends.pct_change() * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    df_norm = (df_trends / df_trends.iloc[0]) * 100
    for col in price_cols:
        ax.plot(df_norm.index, df_norm[col], label=col)

    ax.set_title(f'Xu hướng Giá Hàng hóa & Xăng Nội địa ({days} Ngày Gần Nhất, Chuẩn hóa)')
    ax.set_xlabel('Ngày')
    ax.set_ylabel('Giá (Chuẩn hóa, Ngày đầu = 100)')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Giá')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig, df_pct_change.iloc[1:].tail(5)

# -----------------------------------------------------------------------------------
# F. HÀM PHÂN TÍCH YẾU TỐ TÁC ĐỘNG (FEATURE IMPORTANCE)
# -----------------------------------------------------------------------------------
def get_feature_importance(model, feature_names, top_n=10):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        df_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        df_importance = df_importance.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        return df_importance.head(top_n)
    else:
        return None

# -----------------------------------------------------------------------------------
# G. HÀM PHÂN TÍCH GIÁ HIỆN TẠI (PRICE CONTEXT)
# -----------------------------------------------------------------------------------
def get_price_context(df_raw):
    df = df_raw.copy().reset_index(drop=True)
    df.columns = df.columns.astype(str)
    if TARGET_COL not in df.columns or 'date' not in df.columns:
        return {
            'current_price': np.nan,
            'current_date': pd.NaT,
            'last_adj_date': 'N/A',
            'price_at_adj': np.nan,
            'change_from_adj': 0,
            'change_pct_from_adj': 0
        }

    latest_price = float(df[TARGET_COL].iloc[-1])
    latest_date = pd.to_datetime(df['date'].iloc[-1])

    price_changes = df[TARGET_COL].diff()
    adjustment_dates_series = df.loc[(price_changes != 0) & (price_changes.notna()), 'date']

    # prepare comparison logic safely
    if not adjustment_dates_series.empty and pd.to_datetime(adjustment_dates_series.iloc[-1]).date() == latest_date.date():
        adjustment_days_for_comparison = adjustment_dates_series.iloc[:-1]
    else:
        adjustment_days_for_comparison = adjustment_dates_series

    if (not adjustment_days_for_comparison.empty):
        last_adj_date = adjustment_days_for_comparison.iloc[-1]
        price_at_adj = float(df.loc[df['date'] == last_adj_date, TARGET_COL].iloc[0])
        change_from_adj = latest_price - price_at_adj
        change_pct_from_adj = (change_from_adj / price_at_adj) * 100 if price_at_adj != 0 else 0
    else:
        last_adj_date = 'N/A'
        price_at_adj = latest_price
        change_from_adj = 0
        change_pct_from_adj = 0

    return {
        'current_price': latest_price,
        'current_date': latest_date,
        'last_adj_date': last_adj_date,
        'price_at_adj': price_at_adj,
        'change_from_adj': change_from_adj,
        'change_pct_from_adj': change_pct_from_adj
    }

# -----------------------------------------------------------------------------------
# PHẦN CHÍNH CỦA STREAMLIT APP
# -----------------------------------------------------------------------------------
st.set_page_config(page_title="⛽ Dự đoán Giá Xăng RON 95-III", layout="wide")

# Tải và huấn luyện mô hình
model_results, best_model_name, feature_names, scaler, df_raw = load_and_train_model()

if df_raw is None:
    st.stop()

default_values_raw = df_raw.iloc[-1]
price_context = get_price_context(df_raw)

# Sidebar UI
st.sidebar.header("🔧 Cấu hình Mô hình & Đầu vào")

# Bảng so sánh RMSE
st.sidebar.subheader("📊 Hiệu suất Mô hình (RMSE - VND)")
if model_results:
    rmse_data = {
        'Mô hình': list(model_results.keys()),
        'RMSE (VND)': [f"{model_results[name]['rmse']:,.0f}" for name in model_results.keys()]
    }
    rmse_df = pd.DataFrame(rmse_data)
    st.sidebar.dataframe(rmse_df.set_index('Mô hình'), use_container_width=True)
else:
    st.sidebar.write("Chưa có kết quả mô hình.")

model_selection = st.sidebar.selectbox(
    "Chọn Mô hình Dự đoán",
    options=list(model_results.keys()),
    index=list(model_results.keys()).index(best_model_name) if model_results else 0
)

# Inputs
st.sidebar.subheader("I. Giá Hàng hóa & Tỷ giá (THÔ)")
input_prices = {}
price_fields = [
    ('Gia_Brent(USD)', 'Giá Brent (USD)'),
    ('Gia_WTI(USD)', 'Giá WTI (USD)'),
    ('USD/VND', 'Tỷ giá USD/VND')
]
for feature_name, label in price_fields:
    default_val = float(default_values_raw[feature_name]) if feature_name in default_values_raw and not pd.isna(default_values_raw[feature_name]) else 70.0
    input_prices[feature_name] = st.sidebar.number_input(
        label, value=default_val, step=0.01, format="%.2f", key=f"raw_input_{feature_name}"
    )

st.sidebar.subheader("II. Thông tin Sự kiện")
unique_events = list(EVENT_MAP.keys())
unique_events.insert(0, 'Không có sự kiện')
selected_event = st.sidebar.selectbox("Loại Sự kiện", options=unique_events, index=0)
sentiment = st.sidebar.radio("Xu hướng Sự kiện", options=['None', 'Tăng', 'Giảm'], index=0, disabled=(selected_event == 'Không có sự kiện'))

last_date = pd.to_datetime(df_raw.iloc[-1]['date'])
input_date = st.sidebar.date_input("Ngày Dự đoán (Single-step)", value=last_date + pd.Timedelta(days=1), min_value=last_date + pd.Timedelta(days=1), key="input_date")

# Main UI
st.title("⛽ Ứng dụng Phân tích & Dự đoán Giá Xăng RON 95-III Nội địa")
st.markdown("---")

st.header("🎯 Tóm Tắt & Cảnh Báo")
col1_sum, col2_sum, col3_sum = st.columns(3)

col1_sum.metric("Giá Bán lẻ Hiện tại (RON 95-III)", f"{price_context['current_price']:,.0f} VND", help=f"Giá niêm yết tại ngày cuối cùng của dữ liệu ({price_context['current_date'].strftime('%Y-%m-%d')})")
col2_sum.metric("So sánh với Kỳ điều chỉnh Trước", f"{price_context['change_from_adj']:,.0f} VND", f"{price_context['change_pct_from_adj']:.2f} %", delta_color="inverse", help=f"Thay đổi giá từ ngày điều chỉnh gần nhất ({price_context['last_adj_date']})")

col3_sum.subheader("Cảnh báo")
if col3_sum.button("Kiểm tra cảnh báo", key="check_warning_btn"):
    selected_model = model_results[model_selection]['model']
    raw_input_data = {
        'date': input_date.strftime('%Y-%m-%d'),
        'Gia_Brent(USD)': input_prices['Gia_Brent(USD)'],
        'Gia_WTI(USD)': input_prices['Gia_WTI(USD)'],
        'USD/VND': input_prices['USD/VND'],
        'loai_su_kien': selected_event if selected_event != 'Không có sự kiện' else np.nan,
        'ten_su_kien': np.nan,
        'tang_giam': sentiment if sentiment != 'None' else np.nan,
    }
    try:
        raw_prediction, _ = predict_raw_input(raw_input_data, df_raw, feature_names, scaler, selected_model)
        diff = raw_prediction - price_context['current_price']
        if diff >= 500:
            col3_sum.error(f"⚠️ DỰ BÁO TĂNG MẠNH (Dự kiến: +{diff:,.0f} VND)")
        elif diff <= -500:
            col3_sum.success(f"✅ DỰ BÁO GIẢM MẠNH (Dự kiến: {diff:,.0f} VND)")
        else:
            col3_sum.info("ỔN ĐỊNH: Giá dự kiến thay đổi ít.")
    except Exception as e:
        col3_sum.error(f"Lỗi: {e}")

st.markdown("---")

# PHẦN 1: DỰ ĐOÁN SINGLE-STEP
st.header("1️⃣ Dự đoán Giá xăng Ngày tiếp theo & Phân tích Tác động")
col1_pred, col2_pred = st.columns([1, 1])

if col1_pred.button("Dự đoán Giá Xăng Single-step", type="primary"):
    selected_model = model_results[model_selection]['model']
    raw_input_data = {
        'date': input_date.strftime('%Y-%m-%d'),
        'Gia_Brent(USD)': input_prices['Gia_Brent(USD)'],
        'Gia_WTI(USD)': input_prices['Gia_WTI(USD)'],
        'USD/VND': input_prices['USD/VND'],
        'loai_su_kien': selected_event if selected_event != 'Không có sự kiện' else np.nan,
        'ten_su_kien': np.nan,
        'tang_giam': sentiment if sentiment != 'None' else np.nan,
    }
    try:
        raw_prediction, X_predict = predict_raw_input(raw_input_data, df_raw, feature_names, scaler, selected_model)
        col1_pred.success(f"#### Giá Dự báo ({input_date.strftime('%Y-%m-%d')}): **{raw_prediction:,.0f} VND**")
        df_importance = get_feature_importance(selected_model, feature_names)
        if df_importance is not None:
            col2_pred.subheader("Yếu tố Tác động Lớn nhất")
            col2_pred.dataframe(df_importance.style.format({'Importance': '{:.4f}'}), use_container_width=True)
        else:
            col2_pred.info("Feature Importance chỉ khả dụng cho các mô hình cây (Tree-based Models).")
    except Exception as e:
        col1_pred.error(f"Lỗi trong quá trình dự đoán Single-step: {e}")

st.markdown("---")

# PHẦN 2: DỰ BÁO ĐỆ QUY & CI
st.header("2️⃣ Dự báo Tương lai & Biểu đồ Khoảng tin cậy")
forecast_days_map = {'7 Ngày': 7, '30 Ngày': 30, '90 Ngày': 90}

if st.button("Chạy Dự báo Đệ quy & Khoảng tin cậy", key="run_forecast_btn"):
    st.info("Đang chạy dự báo đệ quy và bootstrap CI...")
    selected_model = model_results[model_selection]['model']
    try:
        # dùng st làm container cho progress bar
        df_forecast = recursive_forecast(df_raw, feature_names, scaler, selected_model, forecast_steps=90, n_bootstraps=30, st_container=st)
        st.subheader("📈 Biểu đồ Dự báo Dài hạn với Khoảng Tin cậy 95%")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_forecast.index, df_forecast['Giá Dự báo'], label='Giá Dự báo')
        ax.fill_between(df_forecast.index, df_forecast['CI 95% Min'], df_forecast['CI 95% Max'], alpha=0.1, label='Khoảng Tin cậy 95%')
        ax.set_title(f"Dự báo Giá RON 95-III - {model_selection}")
        ax.set_xlabel("Ngày")
        ax.set_ylabel("Giá (VND)")
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)

        st.subheader("📝 Tóm tắt Giá Dự báo Ngắn hạn")
        summary_data = []
        for label, days in forecast_days_map.items():
            if days <= len(df_forecast):
                final_row = df_forecast.iloc[days - 1]
                summary_data.append({
                    'Thời gian': label,
                    'Ngày dự báo cuối': df_forecast.index[days - 1].strftime('%Y-%m-%d'),
                    'Giá Dự báo': f"{final_row['Giá Dự báo']:,.0f} VND",
                    'CI 95%': f"[{final_row['CI 95% Min']:,.0f} - {final_row['CI 95% Max']:,.0f}] VND"
                })
        if summary_data:
            st.table(pd.DataFrame(summary_data))
        else:
            st.write("Không đủ bước dự báo để tóm tắt.")
    except Exception as e:
        st.error(f"Lỗi khi chạy dự báo đệ quy: {e}")

st.markdown("---")

# PHẦN 3: BIỂU ĐỒ LỊCH SỬ
st.header("3️⃣ Biến động Giá Lịch sử (6 Tháng / 1 Năm)")
col1_hist, col2_hist = st.columns(2)

if col1_hist.button("Xem Biến động 6 Tháng", key='run_6m_hist_btn'):
    fig_6m, _ = plot_historical_trends(df_raw, days=180)
    col1_hist.subheader("Biến động 6 Tháng")
    col1_hist.pyplot(fig_6m)

if col2_hist.button("Xem Biến động 1 Năm", key='run_1y_hist_btn'):
    fig_1y, _ = plot_historical_trends(df_raw, days=365)
    col2_hist.subheader("Biến động 1 Năm")
    col2_hist.pyplot(fig_1y)
