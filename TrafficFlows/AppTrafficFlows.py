import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from TrafficFlowsModule import V_func, traffic_flow

st.set_page_config(page_title="Traffic Simulator", layout="wide")
st.title("🚗 Traffic Flow Simulation (Ring Road)")

with st.sidebar:
    st.header("Model Parameters")
    N = st.slider("Number of vehicles (N)", 10, 200, 90)
    L = st.number_input("Road length (L)", value=1500)
    T = st.number_input("Total time (T)", value=1000)
    dt = st.number_input("Time step (dt)", value=0.1)

    st.markdown("---")
    st.header("Perturbation & Control")
    pert_time = st.number_input("Perturbation time step", value=100)
    ni = st.number_input("Perturbation factor (ni)", value=0.01)
    alpha = st.number_input("Alpha", value=1.0)
    beta = st.number_input("Beta", value=100.0)
    isheun = st.checkbox("Use Heun's method", value=False)
    isauto = st.checkbox("Active autonomous vehicle", value=False)

if st.button("Run Simulation", type="primary"):
    with st.spinner("Computing equations..."):
        pos = np.linspace(0, L, N, endpoint=False)
        vel = V_func(L / N) * np.ones(N)
        X, V = traffic_flow(
            pos, vel, dt, T, L, N, alpha, beta, ni, pert_time, isheun, isauto
        )

        sampling_step = 50
        X_downsampled = X[::sampling_step, :]
        V_downsampled = V[::sampling_step, :]
        num_frames = X_downsampled.shape[0]

        theta = 2 * np.pi * (X_downsampled / L)
        x_circ = np.cos(theta)
        y_circ = np.sin(theta)

        k_index = 10

        data = []
        for t_idx in range(num_frames):
            real_time = round(t_idx * dt * sampling_step, 1)
            for auto_idx in range(N):
                if isauto:
                    car_type = "Autonomous" if auto_idx == k_index else "Human-driven"
                else:
                    car_type = "Standard"

                data.append(
                    {
                        "Time": real_time,
                        "Car": auto_idx,
                        "X": float(x_circ[t_idx, auto_idx]),
                        "Y": float(y_circ[t_idx, auto_idx]),
                        "Velocity": float(V_downsampled[t_idx, auto_idx]),
                        "Type": car_type,
                    }
                )

        df = pd.DataFrame(data)

    with st.spinner("Rendering animation..."):
        if isauto:
            fig = px.scatter(
                df,
                x="X",
                y="Y",
                animation_frame="Time",
                animation_group="Car",
                color="Type",
                color_discrete_map={"Autonomous": "red", "Human-driven": "lightgrey"},
                range_x=[-1.2, 1.2],
                range_y=[-1.2, 1.2],
                height=700,
            )
        else:
            fig = px.scatter(
                df,
                x="X",
                y="Y",
                animation_frame="Time",
                animation_group="Car",
                color="Velocity",
                color_continuous_scale="Turbo",
                range_color=[0, np.max(V)],
                range_x=[-1.2, 1.2],
                range_y=[-1.2, 1.2],
                height=700,
            )

        fig.update_traces(marker=dict(size=12, line=dict(width=1, color="black")))
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            plot_bgcolor="white",
        )

        st.plotly_chart(fig, use_container_width=True)
