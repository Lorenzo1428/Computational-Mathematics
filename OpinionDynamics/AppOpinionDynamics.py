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
        x = OpinionDynamicsModule.get_init_distributions(dim, choose)
        X, mean = OpinionDynamicsModule.heun_step(x, dt, T, epsi, issym)

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
        st.success("Simulation completed!")
