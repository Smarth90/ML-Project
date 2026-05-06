import requests
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

st.markdown(
    """
    <style>

    .main {
        background-color: #0E1117;
        color: white;
    }

    .stMetric {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 LSTM Stock Prediction Dashboard")

st.write(
    "Deep Learning based stock forecasting using FastAPI + Streamlit"
)


st.sidebar.header("Configuration")

ticker = st.sidebar.text_input(
    "Enter NSE Stock Symbol",
    value="TCS.NS"
)

predict_button = st.sidebar.button(
    "Predict"
)
if predict_button:

    with st.spinner("Fetching predictions..."):

        response = requests.get(
            f"http://127.0.0.1:8000/predict/{ticker}"
        )

        data = response.json()

        actual = np.array(data["actual"]).flatten()

        predicted = np.array(
            data["predicted"]
        ).flatten()

        latest_actual = actual[-1]

        latest_predicted = predicted[-1]

        difference = (
            latest_predicted - latest_actual
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Latest Actual Price",
            f"₹ {latest_actual:.2f}"
        )

        col2.metric(
            "Latest Predicted Price",
            f"₹ {latest_predicted:.2f}"
        )

        col3.metric(
            "Prediction Difference",
            f"₹ {difference:.2f}"
        )
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                y=actual,
                mode='lines',
                name='Actual Price'
            )
        )

        fig.add_trace(
            go.Scatter(
                y=predicted,
                mode='lines',
                name='Predicted Price'
            )
        )

        fig.update_layout(
            title=f"{ticker} Stock Prediction",
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_dark",
            height=600
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.subheader("Prediction Data")

        results_df = pd.DataFrame({
            "Actual": actual,
            "Predicted": predicted
        })

        st.dataframe(
            results_df.tail(20),
            use_container_width=True
        )

        st.success(
            "Prediction completed successfully."
        )