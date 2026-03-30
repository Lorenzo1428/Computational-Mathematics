import matplotlib.pyplot as plt
import OpinionDynamicsModule
import streamlit as st

st.title("Opinion Dynamics Simulation")
st.markdown(
    "Set the parameters in the sidebar and launch the simulation to see how opinions converge!"
)

st.sidebar.header("⚙️ Simulation Parameters")

dim = st.sidebar.number_input(
    "Dimension (Number of agents)", min_value=2, max_value=500, value=10, step=1
)

opzioni_distribuzione = {"Uniform": 1, "Normal": 2, "Pearson": 3}
scelta_nome = st.sidebar.selectbox(
    "Initial Distribution", list(opzioni_distribuzione.keys())
)
choose = opzioni_distribuzione[scelta_nome]

st.sidebar.markdown("---")

epsi = st.sidebar.slider(
    "Epsilon (Confidence radius)",
    min_value=0.01,
    max_value=2.0,
    value=0.2,
    step=0.01,
)
issym = st.sidebar.checkbox("Symmetric Matrix (issym)", value=False)

st.sidebar.markdown("---")

dt = st.sidebar.number_input(
    "Time Step (dt)", min_value=0.01, max_value=1.0, value=0.1, step=0.01
)
T = st.sidebar.number_input(
    "Total Time (T)", min_value=1, max_value=100, value=5, step=1
)

if st.button("🚀 Launch Simulation", type="primary"):
    with st.spinner("Calculating interactions..."):
        # Run simulation
        x = OpinionDynamicsModule.get_init_distributions(dim, choose)
        X, mean = OpinionDynamicsModule.heun_step(x, dt, T, epsi, issym)

        # --- PLOT 1: OPINION EVOLUTION ---
        st.subheader("Trajectories")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(X, alpha=0.8, linewidth=1.5)

        ax.set_title(
            f"Opinion Evolution (Epsilon={epsi}, Symmetric={issym})",
            fontsize=14,
        )
        ax.set_xlabel("Time steps", fontsize=12)
        ax.set_ylabel("Opinion Value", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)

        st.pyplot(fig)

        # --- PLOT 2: STATISTICAL METRICS (MEAN MATRIX) ---
        st.subheader("Statistical Metrics")

        # Create a 1x2 grid: Left for Moments/Variance, Right for Multiplicity
        fig_stats, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig_stats.suptitle("Evolution of Ensemble Metrics", fontsize=16)

        # LEFT PLOT: 1st Moment, 2nd Moment, Variance
        ax1.plot(mean[:, 0], label="1st Moment (Mean)", linewidth=2, color="#1f77b4")
        ax1.plot(mean[:, 1], label="2nd Moment", linewidth=2, color="#ff7f0e")
        ax1.plot(mean[:, 2], label="Variance", linewidth=2, color="#2ca02c")

        ax1.set_title("Moments & Variance")
        ax1.set_xlabel("Time steps")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()  # Aggiunta la legenda per distinguere le tre curve

        # RIGHT PLOT: Multiplicity (Clusters)
        if not issym:
            ax2.plot(
                mean[:, 3],
                color="#d62728",
                linewidth=2,
                drawstyle="steps-post",
                label="Clusters",
            )
        else:
            ax2.plot(mean[:, 3], color="#d62728", linewidth=2, label="Clusters")
            ax2.text(
                0.5,
                0.5,
                "Not calculated for symmetric",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax2.transAxes,
                alpha=0.5,
            )

        ax2.set_title("Multiplicity (Clusters)")
        ax2.set_xlabel("Time steps")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

        plt.tight_layout()
        st.pyplot(fig_stats)

        st.success("Simulation completed!")
