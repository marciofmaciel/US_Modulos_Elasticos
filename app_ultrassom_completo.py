import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import date
import json

# --- Configurações da Página Streamlit ---
st.set_page_config(layout="wide", page_title="Caracterização de Compósitos por Ultrassom")

# --- CSS Personalizado ---
st.markdown("""
    <style>
    /* Estilo para campos de entrada (azul claro) */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background-color: #e0f2f7; /* Light blue */
    }
    /* Estilo para campos de entrada de TOF */
    .tof-input .stNumberInput > div > div > input {
        background-color: #e0f2f7; /* Light blue */
    }
    /* Estilo para resultados de constantes (amarelo claro) */
    .constants-output .stDataFrame {
        background-color: #fffde7; /* Light yellow */
    }
    /* Cores para validação */
    .validation-ok { background-color: #e6ffe6; color: #006600; padding: 5px; border-radius: 3px; } /* Light Green */
    .validation-warning { background-color: #fff8e1; color: #ff8c00; padding: 5px; border-radius: 3px; } /* Light Orange */
    .validation-error { background-color: #ffe6e6; color: #cc0000; padding: 5px; border-radius: 3px; } /* Light Red */
    .validation-global-excellent { color: #006600; font-weight: bold; }
    .validation-global-good { color: #ff8c00; font-weight: bold; }
    .validation-global-poor { color: #cc0000; font-weight: bold; }
    
    /* Ajuste para o título principal */
    h1 {
        color: #2e8b57; /* SeaGreen */
        text-align: center;
    }
    /* Ajuste para subtítulos das abas */
    h2 {
        color: #3cb371; /* MediumSeaGreen */
    }
    /* Ajuste para cabeçalhos de seção */
    h3 {
        color: #4682b4; /* SteelBlue */
    }
    /* Estilo para o DataFrame de constantes */
    .constants-table {
        background-color: #fffde7; /* Light yellow */
        padding: 10px;
        border-radius: 5px;
    }
    /* Estilo para o DataFrame de propriedades */
    .properties-table {
        background-color: #e0f7fa; /* Light cyan */
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 Caracterização de Materiais Compósitos por Ultrassom")

# --- Inicialização do Session State ---
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.material = "Carbono/Epóxi [0/90]₄s"
    st.session_state.codigo = "CE-0904-001"
    st.session_state.data = date.today().strftime("%Y-%m-%d")
    st.session_state.operador = "John Doe"
    st.session_state.espessura = 3.0  # mm
    st.session_state.densidade = 1550.0  # kg/m³
    st.session_state.temp_agua = 20.0 # °C

    # TOF values and their types
    st.session_state.tofs = {
        'C33': {'value': 1.25, 'type': 'Pulse-Echo', 'mode': 'Longitudinal na Direção 3'},
        'C44': {'value': 3.50, 'type': 'Pulse-Echo', 'mode': 'Transversal na Dir. 3, Pol. Dir. 2'},
        'C55': {'value': 3.20, 'type': 'Pulse-Echo', 'mode': 'Transversal na Dir. 3, Pol. Dir. 1'},
        'C11': {'value': 0.50, 'type': 'Pitch-Catch', 'mode': 'Longitudinal na Direção 1'},
        'C22': {'value': 1.00, 'type': 'Pitch-Catch', 'mode': 'Longitudinal na Direção 2'},
        'C66': {'value': 1.50, 'type': 'Pulse-Echo', 'mode': 'Transversal no Plano 1-2'},
        'C12': {'value': 0.75, 'type': 'Imersão Angular', 'mode': 'Quasi-Longitudinal (45° no plano 1-2)'},
        'C13': {'value': 0.80, 'type': 'Imersão Angular', 'mode': 'Quasi-Longitudinal (plano 1-3)'},
        'C23': {'value': 0.95, 'type': 'Imersão Angular', 'mode': 'Quasi-Longitudinal (plano 2-3)'},
    }

# --- Funções de Cálculo (com cache para otimização) ---

@st.cache_data
def calculate_water_velocity(temp_celsius):
    """
    Calcula a velocidade do som na água em m/s com base na temperatura.
    Fórmula aproximada para água doce entre 0-100°C.
    """
    # Fonte: Kaye & Laby, Physical and Chemical Constants, 16th ed., 1995
    # v_w = 1402.39 + 5.0336*T - 0.05810*T^2 + 0.00034*T^3
    return 1402.39 + 5.0336 * temp_celsius - 0.05810 * temp_celsius**2 + 0.00034 * temp_celsius**3

@st.cache_data
def calculate_velocities(h_mm, tofs_data, temp_agua):
    """
    Calcula as velocidades de propagação das ondas.
    h_mm: espessura em mm
    tofs_data: dicionário com TOF e tipo de leitura
    temp_agua: temperatura da água em °C
    Retorna um dicionário de velocidades em m/s.
    """
    h_m = h_mm / 1000.0  # Converter mm para metros
    velocities = {}
    v_agua = calculate_water_velocity(temp_agua)

    for key, data in tofs_data.items():
        tof_us = data['value']
        type_leitura = data['type']
        mode = data['mode']

        if tof_us <= 0:
            velocities[key] = 0.0
            continue

        tof_s = tof_us / 1_000_000.0  # Converter μs para segundos

        if type_leitura == 'Pulse-Echo':
            # v = 2h/t
            v = (2 * h_m) / tof_s
        elif type_leitura == 'Pitch-Catch':
            # v = h/t
            v = h_m / tof_s
        elif type_leitura == 'Imersão Angular':
            # Para imersão angular, a fórmula v = h/t é uma simplificação
            # que assume incidência normal ou correção para ângulo.
            # O protocolo sugere v = h/t para TOF corrigido.
            v = h_m / tof_s
        else:
            v = 0.0 # Tipo de leitura desconhecido

        velocities[key] = v
    return velocities

@st.cache_data
def calculate_elastic_constants(rho_kg_m3, velocities):
    """
    Calcula as constantes elásticas C_ij.
    rho_kg_m3: densidade em kg/m³
    velocities: dicionário de velocidades em m/s
    Retorna um dicionário de constantes em GPa.
    """
    constants = {}
    for key, v in velocities.items():
        if v > 0:
            # C = rho * v^2 (em Pa), depois divide por 1e9 para GPa
            constants[key] = (rho_kg_m3 * (v**2)) / 1e9
        else:
            constants[key] = 0.0
    return constants

@st.cache_data
def build_stiffness_matrix(constants):
    """
    Constrói a matriz de rigidez [C] 6x6 para um material ortotrópico.
    constants: dicionário de constantes C_ij em GPa.
    Retorna a matriz [C] como um array NumPy.
    """
    C_matrix = np.zeros((6, 6))

    # Preencher os elementos da matriz C
    # C11, C22, C33
    C_matrix[0, 0] = constants.get('C11', 0.0)
    C_matrix[1, 1] = constants.get('C22', 0.0)
    C_matrix[2, 2] = constants.get('C33', 0.0)

    # C12, C13, C23 (e suas simétricas)
    C_matrix[0, 1] = C_matrix[1, 0] = constants.get('C12', 0.0)
    C_matrix[0, 2] = C_matrix[2, 0] = constants.get('C13', 0.0)
    C_matrix[1, 2] = C_matrix[2, 1] = constants.get('C23', 0.0)

    # C44, C55, C66 (elementos de cisalhamento)
    C_matrix[3, 3] = constants.get('C44', 0.0)
    C_matrix[4, 4] = constants.get('C55', 0.0)
    C_matrix[5, 5] = constants.get('C66', 0.0)

    return C_matrix

@st.cache_data
def invert_matrix(C_matrix):
    """
    Inverte a matriz de rigidez [C] para obter a matriz de compliance [S].
    C_matrix: matriz [C] como um array NumPy.
    Retorna a matriz [S] como um array NumPy ou None se a inversão falhar.
    """
    try:
        S_matrix = np.linalg.inv(C_matrix)
        return S_matrix
    except np.linalg.LinAlgError:
        return None

@st.cache_data
def calculate_engineering_properties(S_matrix):
    """
    Calcula as propriedades de engenharia a partir da matriz de compliance [S].
    S_matrix: matriz [S] como um array NumPy (em GPa^-1).
    Retorna um dicionário de propriedades.
    """
    properties = {}
    if S_matrix is None:
        return properties

    # Módulos de Young (em GPa)
    properties['E1'] = 1 / S_matrix[0, 0] if S_matrix[0, 0] != 0 else 0
    properties['E2'] = 1 / S_matrix[1, 1] if S_matrix[1, 1] != 0 else 0
    properties['E3'] = 1 / S_matrix[2, 2] if S_matrix[2, 2] != 0 else 0

    # Módulos de Cisalhamento (em GPa)
    properties['G12'] = 1 / S_matrix[5, 5] if S_matrix[5, 5] != 0 else 0
    properties['G13'] = 1 / S_matrix[4, 4] if S_matrix[4, 4] != 0 else 0
    properties['G23'] = 1 / S_matrix[3, 3] if S_matrix[3, 3] != 0 else 0

    # Coeficientes de Poisson
    properties['nu12'] = -S_matrix[0, 1] / S_matrix[0, 0] if S_matrix[0, 0] != 0 else 0
    properties['nu13'] = -S_matrix[0, 2] / S_matrix[0, 0] if S_matrix[0, 0] != 0 else 0
    properties['nu21'] = -S_matrix[0, 1] / S_matrix[1, 1] if S_matrix[1, 1] != 0 else 0 # S12/S22
    properties['nu23'] = -S_matrix[1, 2] / S_matrix[1, 1] if S_matrix[1, 1] != 0 else 0
    properties['nu31'] = -S_matrix[0, 2] / S_matrix[2, 2] if S_matrix[2, 2] != 0 else 0 # S13/S33
    properties['nu32'] = -S_matrix[1, 2] / S_matrix[2, 2] if S_matrix[2, 2] != 0 else 0 # S23/S33

    return properties

@st.cache_data
def validate_data(inputs, velocities, constants, C_matrix, S_matrix, properties):
    """
    Valida os parâmetros críticos e retorna um dicionário de status.
    Retorna um dicionário com resultados de validação e um status global.
    """
    results = {}
    global_status = "Excellent" # Assume excellent initially

    def update_global_status(current_status, new_status):
        if new_status == "ERROR":
            return "Poor"
        elif new_status == "WARNING" and current_status != "Poor":
            return "Good"
        return current_status

    # 1. Espessura > 0
    status = "OK" if inputs['espessura'] > 0 else "ERROR"
    msg = "Espessura deve ser maior que 0."
    results['espessura_valida'] = {'status': status, 'message': msg}
    global_status = update_global_status(global_status, status)

    # 2. Densidade > 0
    status = "OK" if inputs['densidade'] > 0 else "ERROR"
    msg = "Densidade deve ser maior que 0."
    results['densidade_valida'] = {'status': status, 'message': msg}
    global_status = update_global_status(global_status, status)

    # 3. Todos os TOF > 0
    tof_ok = all(data['value'] > 0 for data in inputs['tofs'].values())
    status = "OK" if tof_ok else "ERROR"
    msg = "Todos os Tempos de Voo (TOF) devem ser maiores que 0."
    results['tofs_validos'] = {'status': status, 'message': msg}
    global_status = update_global_status(global_status, status)

    # 4. Velocidades entre 1000-10000 m/s
    vel_ok = True
    vel_warning = False
    for v in velocities.values():
        if not (1000 <= v <= 10000):
            vel_ok = False
            if v < 500 or v > 15000: # Critério mais rigoroso para erro
                status = "ERROR"
                msg = "Algumas velocidades estão fora da faixa esperada (1000-10000 m/s)."
                break
            else:
                vel_warning = True
    if vel_ok:
        status = "OK"
        msg = "Todas as velocidades estão dentro da faixa esperada (1000-10000 m/s)."
    elif vel_warning:
        status = "WARNING"
        msg = "Algumas velocidades estão marginalmente fora da faixa esperada (1000-10000 m/s)."
    else:
        status = "ERROR"
        msg = "Algumas velocidades estão significativamente fora da faixa esperada (1000-10000 m/s)."
    results['velocidades_validas'] = {'status': status, 'message': msg}
    global_status = update_global_status(global_status, status)

    # 5. Todas as constantes C_ij > 0
    const_ok = all(c > 0 for c in constants.values())
    status = "OK" if const_ok else "ERROR"
    msg = "Todas as constantes elásticas C_ij devem ser maiores que 0."
    results['constantes_positivas'] = {'status': status, 'message': msg}
    global_status = update_global_status(global_status, status)

    # 6. Critério: C11*C22 - C12^2 > 0
    c11 = constants.get('C11', 0)
    c22 = constants.get('C22', 0)
    c12 = constants.get('C12', 0)
    criterion_6_value = c11 * c22 - c12**2
    status = "OK" if criterion_6_value > 0 else "ERROR"
    msg = f"Critério C11*C22 - C12^2 > 0. Valor: {criterion_6_value:.2f} GPa²."
    results['c11_c22_c12_crit'] = {'status': status, 'message': msg}
    global_status = update_global_status(global_status, status)

    # 7. Determinante [C] > 0
    if C_matrix is not None:
        try:
            det_C = np.linalg.det(C_matrix)
            status = "OK" if det_C > 0 else "ERROR"
            msg = f"Determinante da matriz [C] > 0. Valor: {det_C:.2e} GPa⁶."
        except np.linalg.LinAlgError:
            status = "ERROR"
            msg = "Não foi possível calcular o determinante da matriz [C]."
    else:
        status = "ERROR"
        msg = "Matriz [C] não pôde ser construída para verificar o determinante."
    results['det_C_positivo'] = {'status': status, 'message': msg}
    global_status = update_global_status(global_status, status)

    # 8. Matriz [C] positiva definida (verificar autovalores)
    if C_matrix is not None:
        try:
            eigenvalues = np.linalg.eigvals(C_matrix)
            positive_definite = all(e > 0 for e in eigenvalues)
            status = "OK" if positive_definite else "ERROR"
            msg = "Matriz [C] é positiva definida (todos os autovalores > 0)." if positive_definite else "Matriz [C] NÃO é positiva definida."
        except np.linalg.LinAlgError:
            status = "ERROR"
            msg = "Não foi possível verificar se a matriz [C] é positiva definida (erro de autovalores)."
    else:
        status = "ERROR"
        msg = "Matriz [C] não pôde ser construída para verificar positividade definida."
    results['C_positive_definite'] = {'status': status, 'message': msg}
    global_status = update_global_status(global_status, status)

    # 9. Módulos de Young positivos
    young_ok = True
    if properties:
        for prop_key in ['E1', 'E2', 'E3']:
            if properties.get(prop_key, 0) <= 0:
                young_ok = False
                break
    else:
        young_ok = False # No properties calculated

    status = "OK" if young_ok else "ERROR"
    msg = "Todos os Módulos de Young (E1, E2, E3) devem ser positivos."
    results['young_moduli_positive'] = {'status': status, 'message': msg}
    global_status = update_global_status(global_status, status)

    # 10. Relações de reciprocidade (ν_ij/E_i = ν_ji/E_j)
    reciprocity_ok = True
    reciprocity_tolerance = 0.05 # 5%
    if properties:
        reciprocity_checks = [
            ('nu12', 'E1', 'nu21', 'E2'),
            ('nu13', 'E1', 'nu31', 'E3'),
            ('nu23', 'E2', 'nu32', 'E3'),
        ]
        for nu_ij_key, E_i_key, nu_ji_key, E_j_key in reciprocity_checks:
            nu_ij = properties.get(nu_ij_key, 0)
            E_i = properties.get(E_i_key, 0)
            nu_ji = properties.get(nu_ji_key, 0)
            E_j = properties.get(E_j_key, 0)

            if E_i != 0 and E_j != 0:
                ratio_ij = nu_ij / E_i
                ratio_ji = nu_ji / E_j
                if ratio_ij != 0: # Avoid division by zero for relative error
                    relative_error = abs(ratio_ij - ratio_ji) / abs(ratio_ij)
                    if relative_error > reciprocity_tolerance:
                        reciprocity_ok = False
                        break
                elif ratio_ji != 0: # If ratio_ij is zero, ratio_ji must also be zero
                    reciprocity_ok = False
                    break
            else: # If any E is zero, reciprocity can't be checked meaningfully, assume not ok
                reciprocity_ok = False
                break
    else:
        reciprocity_ok = False # No properties calculated

    status = "OK" if reciprocity_ok else "WARNING"
    msg = f"Relações de reciprocidade (ν_ij/E_i = ν_ji/E_j) verificadas (tolerância {reciprocity_tolerance*100}%)." if reciprocity_ok else "Relações de reciprocidade NÃO atendidas dentro da tolerância."
    results['reciprocity_valid'] = {'status': status, 'message': msg}
    global_status = update_global_status(global_status, status)

    return results, global_status

# --- Abas da Aplicação ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Entrada de Dados",
    "2. Velocidades",
    "3. Constantes Elásticas",
    "4. Inversão e Propriedades",
    "5. Validação e Alertas",
    "6. Relatório Final"
])

with tab1:
    st.header("📝 Entrada de Dados do Material e Medições")

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.material = st.text_input("Material", value=st.session_state.material, key="input_material")
        st.session_state.codigo = st.text_input("Código", value=st.session_state.codigo, key="input_codigo")
        st.session_state.espessura = st.number_input("Espessura (mm)", min_value=0.001, value=st.session_state.espessura, format="%.3f", key="input_espessura")
        st.session_state.temp_agua = st.number_input("Temperatura da Água (°C)", min_value=0.0, value=st.session_state.temp_agua, format="%.1f", key="input_temp_agua")
    with col2:
        st.session_state.data = st.text_input("Data", value=st.session_state.data, key="input_data")
        st.session_state.operador = st.text_input("Operador", value=st.session_state.operador, key="input_operador")
        st.session_state.densidade = st.number_input("Densidade (kg/m³)", min_value=1.0, value=st.session_state.densidade, format="%.1f", key="input_densidade")

    st.subheader("Tempos de Voo (TOF) em μs")
    st.markdown("---")


    # Grouping TOF inputs by measurement type
    tof_cols = st.columns(3)

    # Pulse-Echo
    with tof_cols[0]:
        st.markdown("### Through-Thickness (Pulse-Echo)")
        for key in ['C33', 'C44', 'C55']:
            st.session_state.tofs[key]['value'] = st.number_input(
                f"TOF {key} ({st.session_state.tofs[key]['mode']}) [μs]",
                min_value=0.0, value=st.session_state.tofs[key]['value'], format="%.2f",
                key=f"tof_input_{key}", help=f"Tipo de leitura: {st.session_state.tofs[key]['type']}"
            )
        st.markdown("### In-Plane (Pulse-Echo)")
        st.session_state.tofs['C66']['value'] = st.number_input(
            f"TOF C66 ({st.session_state.tofs['C66']['mode']}) [μs]",
            min_value=0.0, value=st.session_state.tofs['C66']['value'], format="%.2f",
            key=f"tof_input_C66", help=f"Tipo de leitura: {st.session_state.tofs['C66']['type']}"
        )

    # Pitch-Catch
    with tof_cols[1]:
        st.markdown("### In-Plane (Pitch-Catch)")
        for key in ['C11', 'C22']:
            st.session_state.tofs[key]['value'] = st.number_input(
                f"TOF {key} ({st.session_state.tofs[key]['mode']}) [μs]",
                min_value=0.0, value=st.session_state.tofs[key]['value'], format="%.2f",
                key=f"tof_input_{key}", help=f"Tipo de leitura: {st.session_state.tofs[key]['type']}"
            )
        # Adiciona a imagem após os campos In-Plane (Pitch-Catch)
        st.image("figs/eixos_ultrassom.png", caption="Orientação dos eixos e direções para leituras ultrassônicas")

    # Imersão Angular
    with tof_cols[2]:
        st.markdown("### Acoplamento (Imersão Angular)")
        for key in ['C12', 'C13', 'C23']:
            st.session_state.tofs[key]['value'] = st.number_input(
                f"TOF {key} ({st.session_state.tofs[key]['mode']}) [μs]",
                min_value=0.0, value=st.session_state.tofs[key]['value'], format="%.2f",
                key=f"tof_input_{key}", help=f"Tipo de leitura: {st.session_state.tofs[key]['type']}"
            )

# --- Cálculos Globais (para serem usados em todas as abas) ---
velocities = calculate_velocities(st.session_state.espessura, st.session_state.tofs, st.session_state.temp_agua)
constants = calculate_elastic_constants(st.session_state.densidade, velocities)
C_matrix = build_stiffness_matrix(constants)
S_matrix = invert_matrix(C_matrix)
properties = calculate_engineering_properties(S_matrix)

inputs_for_validation = {
    'espessura': st.session_state.espessura,
    'densidade': st.session_state.densidade,
    'tofs': st.session_state.tofs
}
validation_results, global_validation_status = validate_data(inputs_for_validation, velocities, constants, C_matrix, S_matrix, properties)


with tab2:
    st.header("📐 Cálculo de Velocidades")
    st.write("As velocidades de propagação das ondas são calculadas com base na espessura da amostra e nos tempos de voo (TOF) inseridos.")

    velocities_data = []
    for key, v in velocities.items():
        tof_info = st.session_state.tofs[key]
        formula = "2h/t" if tof_info['type'] == 'Pulse-Echo' else "h/t"
        velocities_data.append({
            "Constante": key,
            "Modo de Onda": tof_info['mode'],
            "Tipo de Leitura": tof_info['type'],
            "TOF (μs)": tof_info['value'],
            "Fórmula Aplicada": formula,
            "Velocidade (m/s)": f"{v:.2f}"
        })
    df_velocities = pd.DataFrame(velocities_data)
    st.dataframe(df_velocities, use_container_width=True, hide_index=True)

with tab3:
    st.header("⚡ Constantes Elásticas C_ij")
    st.write("As constantes elásticas são calculadas a partir da densidade do material e das velocidades de propagação das ondas.")

    constants_data = []
    for key, c in constants.items():
        constants_data.append({
            "Constante": key,
            "Velocidade (m/s)": f"{velocities.get(key, 0):.2f}",
            "Densidade (kg/m³)": st.session_state.densidade,
            "Fórmula": "ρ × v²",
            "Resultado (GPa)": f"{c:.2f}"
        })
    df_constants = pd.DataFrame(constants_data)

    st.markdown('<div class="constants-table">', unsafe_allow_html=True)
    st.subheader("Through-Thickness")
    st.dataframe(df_constants[df_constants['Constante'].isin(['C33', 'C44', 'C55'])], use_container_width=True, hide_index=True)
    st.subheader("In-Plane")
    st.dataframe(df_constants[df_constants['Constante'].isin(['C11', 'C22', 'C66'])], use_container_width=True, hide_index=True)
    st.subheader("Acoplamento")
    st.dataframe(df_constants[df_constants['Constante'].isin(['C12', 'C13', 'C23'])], use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


with tab4:
    st.header("🔄 Inversão e Propriedades de Engenharia")

    st.subheader("Matriz de Rigidez [C] (GPa)")
    if C_matrix is not None:
        df_C = pd.DataFrame(C_matrix, columns=[f"C{i+1}" for i in range(6)], index=[f"C{i+1}" for i in range(6)])
        st.dataframe(df_C.applymap(lambda x: f"{x:.2f}"), use_container_width=True)
    else:
        st.error("Não foi possível construir a Matriz de Rigidez [C]. Verifique os dados de entrada.")

    st.subheader("Matriz de Compliance [S] (GPa⁻¹)")
    if S_matrix is not None:
        df_S = pd.DataFrame(S_matrix, columns=[f"S{i+1}" for i in range(6)], index=[f"S{i+1}" for i in range(6)])
        st.dataframe(df_S.applymap(lambda x: f"{x:.4f}"), use_container_width=True)
    else:
        st.warning("Não foi possível inverter a Matriz de Rigidez [C] para obter a Matriz de Compliance [S]. Verifique os dados de entrada.")

    st.subheader("Propriedades de Engenharia")
    if properties:
        properties_data = [
            {"Propriedade": "E₁", "Fórmula": "1/S₁₁", "Valor": f"{properties.get('E1', 0):.2f}", "Unidade": "GPa"},
            {"Propriedade": "E₂", "Fórmula": "1/S₂₂", "Valor": f"{properties.get('E2', 0):.2f}", "Unidade": "GPa"},
            {"Propriedade": "E₃", "Fórmula": "1/S₃₃", "Valor": f"{properties.get('E3', 0):.2f}", "Unidade": "GPa"},
            {"Propriedade": "G₁₂", "Fórmula": "1/S₆₆", "Valor": f"{properties.get('G12', 0):.2f}", "Unidade": "GPa"},
            {"Propriedade": "G₁₃", "Fórmula": "1/S₅₅", "Valor": f"{properties.get('G13', 0):.2f}", "Unidade": "GPa"},
            {"Propriedade": "G₂₃", "Fórmula": "1/S₄₄", "Valor": f"{properties.get('G23', 0):.2f}", "Unidade": "GPa"},
            {"Propriedade": "ν₁₂", "Fórmula": "-S₁₂/S₁₁", "Valor": f"{properties.get('nu12', 0):.3f}", "Unidade": "-"},
            {"Propriedade": "ν₁₃", "Fórmula": "-S₁₃/S₁₁", "Valor": f"{properties.get('nu13', 0):.3f}", "Unidade": "-"},
            {"Propriedade": "ν₂₁", "Fórmula": "-S₁₂/S₂₂", "Valor": f"{properties.get('nu21', 0):.3f}", "Unidade": "-"},
            {"Propriedade": "ν₂₃", "Fórmula": "-S₂₃/S₂₂", "Valor": f"{properties.get('nu23', 0):.3f}", "Unidade": "-"},
            {"Propriedade": "ν₃₁", "Fórmula": "-S₁₃/S₃₃", "Valor": f"{properties.get('nu31', 0):.3f}", "Unidade": "-"},
            {"Propriedade": "ν₃₂", "Fórmula": "-S₂₃/S₃₃", "Valor": f"{properties.get('nu32', 0):.3f}", "Unidade": "-"},
        ]
        df_properties = pd.DataFrame(properties_data)
        st.markdown('<div class="properties-table">', unsafe_allow_html=True)
        st.dataframe(df_properties, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Não foi possível calcular as propriedades de engenharia. Verifique a inversão da matriz [C].")

with tab5:
    st.header("✔️ Validação e Alertas")
    st.write("Verificação de consistência física e termodinâmica dos dados e resultados.")

    status_map = {
        "OK": "🟢 OK",
        "WARNING": "🟡 Aviso",
        "ERROR": "🔴 Erro"
    }
    css_class_map = {
        "OK": "validation-ok",
        "WARNING": "validation-warning",
        "ERROR": "validation-error"
    }

    for key, result in validation_results.items():
        st.markdown(f'<div class="{css_class_map[result["status"]]}">**{status_map[result["status"]]}**: {result["message"]}</div>', unsafe_allow_html=True)
        st.markdown("") # Add a small space

    st.markdown("---")
    global_status_class = ""
    if global_validation_status == "Excellent":
        global_status_class = "validation-global-excellent"
    elif global_validation_status == "Good":
        global_status_class = "validation-global-good"
    else:
        global_status_class = "validation-global-poor"

    st.markdown(f"### Status Global de Qualidade: <span class='{global_status_class}'>{global_validation_status}</span>", unsafe_allow_html=True)


with tab6:
    st.header("📋 Relatório Final")
    st.write("Resumo completo dos dados de entrada, cálculos e resultados.")

    st.subheader("Dados de Entrada")
    st.markdown(f"""
    - **Material**: {st.session_state.material}
    - **Código**: {st.session_state.codigo}
    - **Data**: {st.session_state.data}
    - **Operador**: {st.session_state.operador}
    - **Espessura**: {st.session_state.espessura:.3f} mm
    - **Densidade**: {st.session_state.densidade:.1f} kg/m³
    - **Temperatura da Água**: {st.session_state.temp_agua:.1f} °C
    """)

    st.subheader("Tempos de Voo (TOF) e Tipos de Leitura")
    tof_report_data = []
    for key, data in st.session_state.tofs.items():
        tof_report_data.append({
            "Constante": key,
            "Modo de Onda": data['mode'],
            "Tipo de Leitura": data['type'],
            "TOF (μs)": f"{data['value']:.2f}"
        })
    df_tof_report = pd.DataFrame(tof_report_data)
    st.dataframe(df_tof_report, use_container_width=True, hide_index=True)

    st.subheader("Velocidades de Propagação")
    st.dataframe(df_velocities, use_container_width=True, hide_index=True)

    st.subheader("Constantes Elásticas C_ij (GPa)")
    st.dataframe(df_constants, use_container_width=True, hide_index=True)

    st.subheader("Propriedades de Engenharia")
    if properties:
        st.dataframe(df_properties, use_container_width=True, hide_index=True)
    else:
        st.warning("Propriedades de engenharia não calculadas devido a erros anteriores.")

    st.subheader("Gráficos dos Módulos")

    # Gráfico de Módulos de Young
    if properties and all(p in properties for p in ['E1', 'E2', 'E3']):
        young_moduli = [properties['E1'], properties['E2'], properties['E3']]
        young_labels = ['E₁', 'E₂', 'E₃']
        fig_young = go.Figure(data=[go.Bar(x=young_labels, y=young_moduli, marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'])])
        fig_young.update_layout(title='Módulos de Young (GPa)', yaxis_title='Valor (GPa)')
        st.plotly_chart(fig_young, use_container_width=True)
    else:
        st.warning("Não foi possível gerar o gráfico dos Módulos de Young.")

    # Gráfico de Módulos de Cisalhamento
    if properties and all(p in properties for p in ['G12', 'G13', 'G23']):
        shear_moduli = [properties['G12'], properties['G13'], properties['G23']]
        shear_labels = ['G₁₂', 'G₁₃', 'G₂₃']
        fig_shear = go.Figure(data=[go.Bar(x=shear_labels, y=shear_moduli, marker_color=['#d62728', '#9467bd', '#8c564b'])])
        fig_shear.update_layout(title='Módulos de Cisalhamento (GPa)', yaxis_title='Valor (GPa)')
        st.plotly_chart(fig_shear, use_container_width=True)
    else:
        st.warning("Não foi possível gerar o gráfico dos Módulos de Cisalhamento.")

    st.subheader("Status de Validação")
    global_status_class = ""
    if global_validation_status == "Excellent":
        global_status_class = "validation-global-excellent"
    elif global_validation_status == "Good":
        global_status_class = "validation-global-good"
    else:
        global_status_class = "validation-global-poor"
    st.markdown(f"### Status Global de Qualidade: <span class='{global_status_class}'>{global_validation_status}</span>", unsafe_allow_html=True)
    for key, result in validation_results.items():
        st.markdown(f'<div class="{css_class_map[result["status"]]}">**{status_map[result["status"]]}**: {result["message"]}</div>', unsafe_allow_html=True)
        st.markdown("") # Add a small space

    st.markdown("---")
    st.subheader("Exportar Dados")

    # Preparar dados para exportação JSON
    export_data = {
        "inputs": {
            "material": st.session_state.material,
            "codigo": st.session_state.codigo,
            "data": st.session_state.data,
            "operador": st.session_state.operador,
            "espessura_mm": st.session_state.espessura,
            "densidade_kg_m3": st.session_state.densidade,
            "temperatura_agua_celsius": st.session_state.temp_agua,
            "tofs_us": st.session_state.tofs
        },
        "velocities_m_s": velocities,
        "elastic_constants_gpa": constants,
        "stiffness_matrix_gpa": C_matrix.tolist() if C_matrix is not None else None,
        "compliance_matrix_gpa_inv": S_matrix.tolist() if S_matrix is not None else None,
        "engineering_properties": properties,
        "validation_results": validation_results,
        "global_validation_status": global_validation_status
    }
    json_string = json.dumps(export_data, indent=4)
    st.download_button(
        label="Download Dados (JSON)",
        data=json_string,
        file_name=f"{st.session_state.codigo}_ultrasound_data.json",
        mime="application/json"
    )

    st.info("Para imprimir este relatório, use a função de impressão do seu navegador (Ctrl+P ou Cmd+P).")