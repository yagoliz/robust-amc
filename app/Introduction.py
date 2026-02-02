"""Robust AMC Demonstrator - Streamlit Dashboard.

This is the main entry point for the interactive dashboard.
Run with: streamlit run app/main.py
"""

import streamlit as st

st.set_page_config(
    page_title="Robust AMC Demonstrator",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main page content
st.title("Robust Automatic Modulation Classification")
st.markdown("### Deep Learning for RF Signal Classification")

st.markdown("""
This demonstrator showcases techniques for **robust automatic modulation classification (AMC)**
using deep learning. It is based on research covering:

- **Domain Shift**: How real-world impairments degrade classifier performance
- **Data Augmentation**: Training-time techniques to improve robustness
- **Contrastive Learning**: Self-supervised methods for better feature representations

---

### Features

Navigate using the sidebar to explore:

1. **Signal Explorer** - Visualize I/Q signals and constellation diagrams
2. **Impairment Simulator** - See how hardware impairments affect signals in real-time
3. **Model Evaluation** - Test the classifier on clean and impaired signals
4. **Domain Shift Demo** - Observe accuracy collapse under various conditions

---

### Dataset

The demonstrator uses **RadioML2016.10a**:
- 11 modulation classes (BPSK, QPSK, 8PSK, QAM16, QAM64, etc.)
- 220,000 samples at SNRs from -20 to +18 dB
- I/Q format: 128 time samples per signal

### Model

**PF-CNN (Phase-Feature CNN)**: A dual-branch architecture that processes
amplitude and phase information separately before fusion.
""")

# Sidebar info
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    Built with:
    - PyTorch
    - Streamlit
    - RadioML2016.10a
    """)
