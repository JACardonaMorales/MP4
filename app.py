import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# =========================================================
# 1. GENERACIÓN DE ATRIBUTOS (POOL COMPLETO DE VARIABLES)
# =========================================================
@st.cache_data
def preparar_datos_completos():
    df = pd.read_csv("results.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    elo = {}
    team_history = {}
    k = 30
    records = []
    
    def calc_mean(h, limit):
        return sum(h[-limit:]) / min(len(h), limit) if h else 0
        
    for row in df.itertuples():
        home, away = row.home_team, row.away_team
        
        # 1. Elo Rating
        r_home = elo.get(home, 1500)
        r_away = elo.get(away, 1500)
        h_adv = 100 if not row.neutral else 0
        diff_elo = (r_home + h_adv) - r_away
        
        if home not in team_history: team_history[home] = {'res': [], 'gf': [], 'ga': []}
        if away not in team_history: team_history[away] = {'res': [], 'gf': [], 'ga': []}
        
        # 2, 3, 4. Métricas de forma y goles
        diff_form = calc_mean(team_history[home]['res'], 5) - calc_mean(team_history[away]['res'], 5)
        diff_gf = calc_mean(team_history[home]['gf'], 10) - calc_mean(team_history[away]['gf'], 10)
        diff_ga = calc_mean(team_history[home]['ga'], 10) - calc_mean(team_history[away]['ga'], 10)
        
        # 5, 6. Binarias
        is_neutral = int(row.neutral)
        is_tournament = int(row.tournament != 'Friendly')
        
        if row.home_score > row.away_score:
            res_h, res_a, y_val = 1.0, 0.0, 0
        elif row.home_score == row.away_score:
            res_h, res_a, y_val = 0.5, 0.5, 1
        else:
            res_h, res_a, y_val = 0.0, 1.0, 2
            
        if row.date.year >= 1990:
            records.append({
                'diff_elo': diff_elo,
                'diff_form': diff_form,
                'diff_gf': diff_gf,
                'diff_ga': diff_ga,
                'is_neutral': is_neutral,
                'is_tournament': is_tournament,
                'y': y_val
            })
            
        exp_home = 1.0 / (1.0 + 10 ** (-diff_elo / 400.0))
        exp_away = 1.0 - exp_home
        elo[home] = r_home + k * (res_h - exp_home)
        elo[away] = r_away + k * (res_a - exp_away)
        
        team_history[home]['res'].append(1 if res_h == 1.0 else 0)
        team_history[away]['res'].append(1 if res_a == 1.0 else 0)
        team_history[home]['gf'].append(row.home_score)
        team_history[away]['gf'].append(row.away_score)
        team_history[home]['ga'].append(row.away_score)
        team_history[away]['ga'].append(row.home_score)

    return pd.DataFrame(records)

# =========================================================
# 2. DEFINICIÓN DEL MODELO MLP Y ENTRENAMIENTO
# =========================================================
def softmax(z):
    exps = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

class MLP:
    def __init__(self, layer_sizes, activation_name, lr):
        self.sizes = layer_sizes
        self.activations = [activation_name] * (len(layer_sizes) - 2) + ['softmax']
        self.lr = lr
        np.random.seed(42)
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1 for i in range(len(layer_sizes)-1)]
        self.biases = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]
        self.num_params = sum(w.size + b.size for w, b in zip(self.weights, self.biases))
        
    def forward(self, X):
        self.A = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.A[-1], self.weights[i]) + self.biases[i]
            if self.activations[i] == 'relu': a = np.maximum(0, z)
            elif self.activations[i] == 'sigmoid': a = 1 / (1 + np.exp(-np.clip(z, -250, 250)))
            elif self.activations[i] == 'tanh': a = np.tanh(z)
            else: a = softmax(z)
            self.A.append(a)
        return self.A[-1]
        
    def backward(self, dZ):
        m = dZ.shape[0]
        for i in reversed(range(len(self.weights))):
            dW = np.dot(self.A[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
                act_name = self.activations[i-1]
                if act_name == 'relu': dZ = dA * (self.A[i] > 0).astype(float)
                elif act_name == 'sigmoid': dZ = dA * (self.A[i] * (1 - self.A[i]))
                elif act_name == 'tanh': dZ = dA * (1 - self.A[i]**2)
            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db

# Función con recopilación de pérdidas y barra de progreso
def entrenar_y_evaluar(df_data, feat_list, act_func, lr, epochs, batch_size, neuronas, show_progress=False):
    X = df_data[feat_list].values.astype(float)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    y = df_data['y'].values
    Y_onehot = np.zeros((y.size, 3))
    Y_onehot[np.arange(y.size), y] = 1
    
    layer_sizes = [len(feat_list)] + neuronas + [3]
    mlp = MLP(layer_sizes, act_func, lr)
    pesos_clases = np.array([1.0, 1.25, 1.0])
    
    progress_bar = st.progress(0) if show_progress else None
    status_text = st.empty() if show_progress else None
    losses = []
    stopped_epoch = None
    
    t0 = time.time()
    for e in range(epochs):
        indices = np.random.permutation(len(X))
        X_shuff, Y_shuff = X[indices], Y_onehot[indices]
        
        batch_losses = []
        for i in range(0, len(X), batch_size):
            X_b, Y_b = X_shuff[i:i+batch_size], Y_shuff[i:i+batch_size]
            pred = mlp.forward(X_b)
            
            w_b = np.sum(Y_b * pesos_clases, axis=1, keepdims=True)
            pred_clip = np.clip(pred, 1e-15, 1 - 1e-15)
            batch_losses.append(-np.sum(w_b * Y_b * np.log(pred_clip)) / len(Y_b))
            
            dZ = (pred - Y_b) * w_b
            mlp.backward(dZ)
            
        epoch_loss = np.mean(batch_losses)
        losses.append(epoch_loss)
        
        if show_progress:
            progress_bar.progress((e + 1) / epochs)
            status_text.text(f"Época {e+1}/{epochs} | Loss: {epoch_loss:.4f}")
            
        if e > 20 and abs(losses[-1] - losses[-2]) < 1e-6:
            stopped_epoch = e + 1
            break
            
    t_comp = time.time() - t0
    
    preds = np.argmax(mlp.forward(X), axis=1)
    acc = np.mean(y == preds)
    
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y, preds): cm[t, p] += 1
    f1s = []
    for i in range(3):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0)
        
    return acc, np.mean(f1s), t_comp, mlp.num_params, cm, losses, stopped_epoch

# =========================================================
# 3. INTERFAZ DE USUARIO (Streamlit)
# =========================================================
st.set_page_config(page_title="Predicción de Fútbol ANN", layout="wide")
st.title("⚽ Clasificador Multicategórico de Fútbol (Red Neuronal)")

# Carga y preparación de los datos
df_full = preparar_datos_completos()

todos_los_atributos = ['diff_elo', 'diff_form', 'diff_gf', 'diff_ga', 'is_neutral', 'is_tournament']

# Barra lateral: Selección de Atributos
st.sidebar.header("1. Selección de Atributos")
atributos_sel = st.sidebar.multiselect(
    "Atributos de entrada:",
    options=todos_los_atributos,
    default=['diff_elo', 'diff_form', 'diff_gf', 'diff_ga', 'is_neutral']
)

# Barra lateral: Hiperparámetros
st.sidebar.header("2. Hiperparámetros de la Red")
n_capas = st.sidebar.slider("Capas Ocultas", 1, 2, 1)
neuronas = [st.sidebar.slider(f"Neuronas Capa Oculta {i+1}", 4, 8, 4) for i in range(n_capas)]
act_func = st.sidebar.selectbox("Función de Activación", ["relu", "sigmoid", "tanh"])
lr = st.sidebar.slider("Tasa de Aprendizaje (η)", 0.01, 1.0, 0.1, step=0.01)
epochs = st.sidebar.slider("Épocas Máximas", 10, 500, 50)
batch_size = st.sidebar.select_slider("Batch Size", options=[32, 64, 128, 256, 512, len(df_full)], value=256)

tab1, tab2 = st.tabs(["🚀 Entrenamiento Individual", "📊 Comparativa Automática de Atributos"])

with tab1:
    if len(atributos_sel) == 0:
        st.warning("⚠️ Debes seleccionar al menos un atributo en la barra lateral.")
    else:
        st.write(f"**Atributos seleccionados ({len(atributos_sel)}):** `{', '.join(atributos_sel)}`")
        if st.button("Entrenar Modelo con Atributos Seleccionados", type="primary"):
            acc, f1, t_comp, params, cm, losses, stopped_epoch = entrenar_y_evaluar(
                df_full, atributos_sel, act_func, lr, epochs, batch_size, neuronas, show_progress=True
            )
            
            if stopped_epoch:
                st.warning(f"Entrenamiento detenido en la época {stopped_epoch} por convergencia temprana.")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc*100:.2f}%")
            c2.metric("F1-Score (Macro)", f"{f1:.4f}")
            c3.metric("Tiempo Cómputo", f"{t_comp:.2f} s")
            c4.metric("Complejidad (Pesos+Sesgos)", params)
            
            # Gráficas en 2 columnas: Costo y Matriz de Confusión
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # 1. Curva de Pérdida
            ax1.plot(losses, color='blue', linewidth=2)
            ax1.set_title("Evolución del Costo (Cross-Entropy Ponderado)")
            ax1.set_xlabel("Épocas")
            ax1.set_ylabel("Pérdida")
            ax1.grid(True)
            
            # 2. Matriz de Confusión
            cax = ax2.matshow(cm, cmap='Blues')
            fig.colorbar(cax)
            for (i, j), val in np.ndenumerate(cm):
                ax2.text(j, i, f'{val}', ha='center', va='center', color='red' if val > cm.max()/2 else 'black')
            ax2.set_title("Matriz de Confusión (0=Local, 1=Empate, 2=Visita)")
            ax2.set_xlabel("Predicción")
            ax2.set_ylabel("Real")
            
            plt.tight_layout()
            st.pyplot(fig)

with tab2:
    st.subheader("Comparativa de Atributos Solos vs Combinaciones")
    st.write("Presiona el botón para entrenar y evaluar automáticamente el impacto de cada atributo individual y de las combinaciones más relevantes.")
    
    if st.button("Ejecutar Comparativa Completa"):
        with st.spinner("Ejecutando pruebas y calculando métricas..."):
            experimentos = []
            
            for f in todos_los_atributos:
                experimentos.append({'Nombre': f"Solo {f}", 'Features': [f]})
                
            experimentos.append({
                'Nombre': "Propuesta original guía (5 atributos)", 
                'Features': ['diff_form', 'diff_gf', 'diff_ga', 'is_neutral', 'is_tournament']
            })
            experimentos.append({
                'Nombre': "Top 5 (con diff_elo)", 
                'Features': ['diff_elo', 'diff_form', 'diff_gf', 'diff_ga', 'is_neutral']
            })
            experimentos.append({
                'Nombre': "Mínima jerárquica (Elo + Localía)", 
                'Features': ['diff_elo', 'is_neutral']
            })
            experimentos.append({
                'Nombre': "Elo + Goles", 
                'Features': ['diff_elo', 'diff_gf', 'diff_ga']
            })
            
            filas = []
            for exp in experimentos:
                acc_e, f1_e, t_e, p_e, _, _, _ = entrenar_y_evaluar(
                    df_full, exp['Features'], 'relu', 0.1, 30, 256, [4], show_progress=False
                )
                filas.append({
                    'Configuración / Atributo(s)': exp['Nombre'],
                    'Cant. Atributos': len(exp['Features']),
                    'Accuracy': f"{acc_e*100:.2f}%",
                    'F1-Macro': round(f1_e, 4),
                    'Complejidad (Pesos+Sesgos)': p_e,
                    'Tiempo (s)': round(t_e, 2)
                })
                
            tabla_res = pd.DataFrame(filas)
            st.dataframe(tabla_res, use_container_width=True)
            
            fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
            nombres = [f['Configuración / Atributo(s)'] for f in filas]
            valores_acc = [float(f['Accuracy'].replace('%', '')) for f in filas]
            ax_bar.barh(nombres, valores_acc, color='#0284c7')
            ax_bar.set_xlabel("Accuracy (%)")
            ax_bar.set_title("Comparativa de Rendimiento por Atributo y Combinación")
            ax_bar.set_xlim(40, 65)
            for i, v in enumerate(valores_acc):
                ax_bar.text(v + 0.5, i, f"{v:.2f}%", va='center', fontweight='bold')
            st.pyplot(fig_bar)