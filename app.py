
import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go

# --- PAGE CONFIG & CSS ---
st.set_page_config(page_title="Pro Linear Algebra Tutor", layout="wide")

# CSS to enable horizontal scrolling for large matrices
st.markdown("""
    <style>
    .math-container {
        overflow-x: auto;
        padding-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SYMBOLIC SETUP ---
x_sym = sp.symbols('x')

# --- SIDEBAR: GLOBAL CONTROLS ---
st.sidebar.header("🕹️ Global Controls")
n = st.sidebar.number_input("Matrix Dimension (n x n)", min_value=2, max_value=10, value=3)


def get_rand_mat(dim, m_type):
    """Generates a random integer matrix."""
    evals = [np.random.randint(-3, 4) for _ in range(dim)]
    if m_type == "Singular":
        evals[0] = 0
    else:
        evals = [v if v != 0 else 1 for v in evals]

    P = sp.Matrix(np.random.randint(-1, 2, size=(dim, dim)))
    while P.det() == 0:
        P = sp.Matrix(np.random.randint(-1, 2, size=(dim, dim)))
    return P * sp.diag(*evals) * P.inv()


# --- SESSION GUARD ---
# Prevents dimensionality crashes when slider changes
if "current_n" not in st.session_state or st.session_state.current_n != n:
    st.session_state.current_n = n
    st.session_state.matrix_A = sp.eye(n)
    st.session_state.matrix_B = sp.eye(n)
    st.rerun()

st.title("🔢 High-Dimensional Matrix & Calculus Tutor")

# --- TABS ---
tab_build, tab_calc, tab_eigen, tab_viz = st.tabs([
    "🏗️ Matrix Construction",
    "🧮 Operations & Calculus",
    "🧬 Eigen-Analysis"
])

# ==========================================
# TAB 1: BUILD & SAVE
# ==========================================
with tab_build:
    col_a, col_b = st.columns(2)

    for label, col in zip(['A', 'B'], [col_a, col_b]):
        with col:
            st.subheader(f"Matrix {label}")
            mode = st.radio(f"Mode {label}", ["Manual", "Auto"], key=f"m_{label}", horizontal=True)

            if mode == "Auto":
                m_t = st.selectbox(f"Type {label}", ["Non-Singular", "Singular"], key=f"t_{label}")
                if st.button(f"Generate {label}"):
                    st.session_state[f'matrix_{label}'] = get_rand_mat(n, m_t)

            # Dynamic Grid
            mat = st.session_state[f'matrix_{label}']
            rows_data = []
            for r in range(n):
                grid = st.columns(n)
                row_cells = []
                for c in range(n):
                    cell_val = grid[c].text_input(f"{label}{r + 1}{c + 1}", value=str(mat[r, c]),
                                                  key=f"c_{label}_{r}_{c}")
                    try:
                        row_cells.append(sp.sympify(cell_val))
                    except:
                        row_cells.append(sp.core.numbers.Zero())  # Default to 0 if invalid
                rows_data.append(row_cells)

            if st.button(f"Save Matrix {label}", key=f"save_{label}"):
                st.session_state[f'matrix_{label}'] = sp.Matrix(rows_data)
                st.success(f"Matrix {label} Updated!")

# ==========================================
# TAB 2: OPERATIONS & CALCULUS
# ==========================================
with tab_calc:
    col_arith, col_inv, col_der = st.columns([1, 1, 1])

    with col_arith:
        st.subheader("🛠️ Arithmetic")
        op = st.selectbox("Operation:", ["A + B", "A - B", "A * B (Dot)"], key="op_select")

        A = st.session_state.matrix_A
        B = st.session_state.matrix_B
        result = A + B if op == "A + B" else A - B if op == "A - B" else A * B

        st.markdown('<div class="math-container">', unsafe_allow_html=True)
        st.latex(sp.latex(result))
        st.markdown('</div>', unsafe_allow_html=True)

    with col_inv:
        st.subheader("🔄 Inversion")
        inv_target = st.radio("Invert:", ["None", "Matrix A", "Matrix B"], horizontal=True)

        if inv_target != "None":
            target_mat = A if inv_target == "Matrix A" else B
            try:
                if target_mat.det() != 0:
                    st.markdown('<div class="math-container">', unsafe_allow_html=True)
                    st.latex(rf"{inv_target}^{{-1}} = {sp.latex(target_mat.inv())}")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error(f"{inv_target} is Singular (Determinant = 0)")
            except Exception as e:
                st.error("Cannot compute determinant. Contains unresolved symbols?")

    with col_der:
        st.subheader("📉 Calculus Suite")
        calc_target = st.radio("Apply to:", ["A", "B"], key="calc_target", horizontal=True)
        M_calc = A if calc_target == "A" else B

        c_btn1, c_btn2 = st.columns(2)
        show_diff = c_btn1.button("d/dx")
        show_int = c_btn2.button("∫ dx")

        if show_diff:
            st.write(f"Derivative of Matrix {calc_target}:")
            st.markdown('<div class="math-container">', unsafe_allow_html=True)
            st.latex(sp.latex(M_calc.diff(x_sym)))
            st.markdown('</div>', unsafe_allow_html=True)

        if show_int:
            st.write(f"Integral of Matrix {calc_target}:")
            st.markdown('<div class="math-container">', unsafe_allow_html=True)
            st.latex(rf"{sp.latex(M_calc.integrate(x_sym))} + \mathcal{{C}}")
            st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 3: EIGEN-ANALYSIS
# ==========================================
with tab_eigen:
    target = st.selectbox("Analyze Matrix:", ["A", "B"], key="eigen_target")
    M = st.session_state.matrix_A if target == "A" else st.session_state.matrix_B

    st.write(f"### Spectral Analysis of Matrix {target}")

    if st.button("Compute Eigen-System"):
        try:
            # eigenvects returns: (eigenvalue, multiplicity, [eigenvectors])
            e_system = M.eigenvects()

            st.write("#### Results:")
            for val, mult, vecs in e_system:
                ec1, ec2 = st.columns([1, 2])
                with ec1:
                    st.info(f"**Eigenvalue ($\lambda$):**")
                    st.latex(sp.latex(val))
                    st.write(f"*Algebraic Multiplicity:* {mult}")
                with ec2:
                    st.write("**Corresponding Eigenvector(s):**")
                    st.markdown('<div class="math-container">', unsafe_allow_html=True)
                    for v in vecs:
                        st.latex(sp.latex(v))
                    st.markdown('</div>', unsafe_allow_html=True)
                st.divider()
        except Exception as e:
            st.error("Computation too complex! Try a smaller matrix or purely numeric values.")

