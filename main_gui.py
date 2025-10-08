import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Config visual
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


class F1AnalyzerGUI:
    def __init__(self, root):
        """Inicializa la interfaz gráfica de usuario."""
        self.root = root
        self.root.title("Analizador de Rendimiento de Formula 1")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1a1a1a")

        # Variables de datos (inicialmente vacías para evitar errores si faltan archivos)
        self.data_path = os.path.join(os.getcwd(), "Data")
        self.drivers = pd.DataFrame()
        self.races = pd.DataFrame()
        self.results = pd.DataFrame()
        self.lap_times = pd.DataFrame()
        self.constructors = pd.DataFrame()
        self.df_clean = pd.DataFrame()

        # Variables del modelo
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.mse = None
        self.r2 = None

        # Cargar y preparar datos / modelo
        self.load_data()
        self.train_prediction_model()

        # Crear la interfaz de usuario
        self.create_widgets()

    # ---------------- Data loading ----------------
    def load_data(self):
        """Carga y limpia los datos de las carreras de F1 si existen."""
        files = {
            'drivers': "drivers.csv",
            'races': "races.csv",
            'results': "results.csv",
            'lap_times': "lap_times.csv",
            'constructors': "constructors.csv",
            'df_clean': "f1_clean_Dataset.csv"  # dataset preparado (opcional)
        }

        for attr, fname in files.items():
            path = os.path.join(self.data_path, fname)
            try:
                setattr(self, attr if attr != 'df_clean' else 'df_clean', pd.read_csv(path))
            except FileNotFoundError:
                # No abortamos: dejamos DataFrame vacío y mostramos advertencia
                messagebox.showwarning("Archivo no encontrado",
                                       f"No se encontró '{fname}' en '{self.data_path}'. Algunas funciones quedarán limitadas.")
                setattr(self, attr if attr != 'df_clean' else 'df_clean', pd.DataFrame())
            except Exception as e:
                messagebox.showerror("Error al cargar datos", f"Error leyendo {fname}:\n{e}")
                setattr(self, attr if attr != 'df_clean' else 'df_clean', pd.DataFrame())

    # ---------------- Model training ----------------
    def train_prediction_model(self):
        """Entrena el modelo de predicción de puntos si hay dataset limpio disponible."""
        if self.df_clean is None or self.df_clean.empty:
            # Intentar usar results si df_clean no está disponible
            self.model = None
            return

        try:
            df = self.df_clean.replace('\\N', pd.NA)
            # Selección segura de columnas
            needed = ['points', 'position', 'grid', 'laps', 'constructorId', 'circuitId', 'driverId']
            if not all(col in df.columns for col in needed):
                # dataset no tiene las columnas requeridas
                self.model = None
                return

            data = df[needed].dropna()
            x = data[['position', 'grid', 'laps', 'constructorId', 'circuitId', 'driverId']]
            y = data['points']

            X = pd.get_dummies(x.astype(object), columns=['constructorId', 'circuitId', 'driverId'], drop_first=True)

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2,
                                                                                    random_state=42)
            self.model = LinearRegression()
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)
            self.mse = mean_squared_error(self.y_test, self.y_pred)
            self.r2 = r2_score(self.y_test, self.y_pred)
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar el modelo:\n{e}")
            self.model = None

    # ---------------- UI creation ----------------
    def create_widgets(self):
        """Crea los widgets de la interfaz gráfica de usuario."""

        # Header
        header_frame = tk.Frame(self.root, bg="#e10600", height=80)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="Analizador de Rendimiento de Formula 1",
            font=("Arial", 24, "bold"),
            bg="#e10600",
            fg="white"
        )
        title_label.pack(expand=True)

        # Panel izquierdo
        left_panel = tk.Frame(self.root, bg="#2d2d2d", width=300)
        left_panel.pack(fill=tk.Y, side=tk.LEFT)
        left_panel.pack_propagate(False)

        # Título del panel
        panel_title = tk.Label(
            left_panel,
            text="ANÁLISIS DISPONIBLES",
            font=("Arial", 14, "bold"),
            bg="#2d2d2d",
            fg="white",
            pady=20
        )
        panel_title.pack()

        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, padx=10, pady=5)

        # Botones - Pilotos
        self.create_section_label(left_panel, "PILOTOS")
        self.create_button(left_panel, "Top Victorias", self.show_top_winners)
        self.create_button(left_panel, "Top Podios", self.show_top_podiums)
        self.create_button(left_panel, "Top Puntos", self.show_top_points)
        self.create_button(left_panel, "Buscar Piloto", self.search_driver)

        ttk.Separator(left_panel, orient="horizontal").pack(fill=tk.X, padx=10, pady=10)

        # Botones - Equipos
        self.create_section_label(left_panel, "EQUIPOS")
        self.create_button(left_panel, "Top Victorias", self.show_top_constructor_wins)
        self.create_button(left_panel, "Top Puntos", self.show_top_constructor_points)
        self.create_button(left_panel, "Evolución", self.show_constructor_evolution)

        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, padx=10, pady=10)

        # Comparaciones
        self.create_section_label(left_panel, "COMPARACIONES")
        self.create_button(left_panel, "Comparar pilotos", self.compare_drivers)

        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, padx=10, pady=10)

        # Predicción
        self.create_section_label(left_panel, "PREDICCIÓN ML")
        self.create_button(left_panel, "Modelo de predicción", self.show_model_performance, bg="#9333ea")
        self.create_button(left_panel, "Predecir Piloto", self.predict_driver_points, bg="#9333ea")

        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, padx=10, pady=10)

        # Visualizaciones
        self.create_section_label(left_panel, "VISUALIZACIONES")
        self.create_button(left_panel, "Mapa de Calor", self.show_heatmap)
        self.create_button(left_panel, "Distribución", self.show_distribution)
        self.create_button(left_panel, "Top 5 Evolución", self.show_top5_evolution)

        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, padx=10, pady=10)

        # Estadísticas generales
        self.create_button(left_panel, "Estadísticas Generales", self.show_general_stats, bg="#00D2BE")

        # Área principal (derecha)
        self.main_area = tk.Frame(self.root, bg="#1a1a1a")
        self.main_area.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)

        # Mensaje de bienvenida
        self.show_welcome_message()

    def create_section_label(self, parent, text):
        """Crea una etiqueta de sección"""
        label = tk.Label(
            parent,
            text=text,
            font=("Arial", 11, "bold"),
            bg="#2d2d2d",
            fg="#00D2BE",
            anchor="w",
            padx=15,
            pady=5
        )
        label.pack(fill=tk.X)

    def create_button(self, parent, text, command, bg="#e10600"):
        """Crea un botón estilizado"""
        button = tk.Button(
            parent,
            text=text,
            command=command,
            font=("Arial", 10),
            bg=bg,
            fg="white",
            activebackground="#ff2020" if bg == "#e10600" else "#a855f7",
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",
            padx=10,
            pady=8
        )
        button.pack(fill=tk.X, padx=15, pady=3)

        # Efecto hover
        hover_color = "#ff2020" if bg == "#e10600" else ("#a855f7" if bg == "#9333ea" else "#00e5d0")
        button.bind("<Enter>", lambda e: button.config(bg=hover_color))
        button.bind("<Leave>", lambda e: button.config(bg=bg))

    def clear_main_area(self):
        """Limpia el área principal"""
        for widget in self.main_area.winfo_children():
            widget.destroy()

    def show_welcome_message(self):
        """Muestra el mensaje de bienvenida"""
        self.clear_main_area()

        welcome_frame = tk.Frame(self.main_area, bg="#1a1a1a")
        welcome_frame.pack(expand=True)

        welcome_label = tk.Label(
            welcome_frame,
            text="🏁 Bienvenido al Analizador de F1 🏁",
            font=("Arial", 28, "bold"),
            bg="#1a1a1a",
            fg="white"
        )
        welcome_label.pack(pady=20)

        subtitle = tk.Label(
            welcome_frame,
            text="Análisis de Datos + Machine Learning",
            font=("Arial", 16),
            bg="#1a1a1a",
            fg="#9333ea"
        )
        subtitle.pack(pady=5)

        subtitle2 = tk.Label(
            welcome_frame,
            text="Selecciona una opción del panel izquierdo para comenzar",
            font=("Arial", 12),
            bg="#1a1a1a",
            fg="#cccccc"
        )
        subtitle2.pack(pady=10)

        model_status = "✓ Modelo entrenado" if self.model is not None else "✗ Modelo no disponible"
        r2_pct = f"{self.r2 * 100:.1f}%" if self.r2 is not None else "N/A"

        info_text = f"""
Dataset cargado (resumen):

• Pilotos: {len(self.drivers)}
• Carreras: {len(self.races)}
• Equipos: {len(self.constructors)}
• Años: {(self.races['year'].min() if not self.races.empty else 'N/A')} - {(self.races['year'].max() if not self.races.empty else 'N/A')}

Modelo de Predicción ML:
• Estado: {model_status}
• Precisión (R²): {r2_pct}
"""
        info_label = tk.Label(
            welcome_frame,
            text=info_text,
            font=("Arial", 12),
            bg="#1a1a1a",
            fg="#00D2BE",
            justify=tk.LEFT
        )
        info_label.pack(pady=30)

    def show_plot_in_main_area(self, fig, title="Resultado"):
        """Muestra un gráfico de matplotlib en el área principal"""
        self.clear_main_area()

        title_label = tk.Label(
            self.main_area,
            text=title,
            font=("Arial", 18, "bold"),
            bg="#1a1a1a",
            fg="white",
            pady=15
        )
        title_label.pack()

        canvas = FigureCanvasTkAgg(fig, master=self.main_area)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    def show_text_result(self, text, title="Resultado"):
        """Muestra texto en el área principal"""
        self.clear_main_area()

        title_label = tk.Label(
            self.main_area,
            text=title,
            font=("Arial", 18, "bold"),
            bg="#1a1a1a",
            fg="white",
            pady=15
        )
        title_label.pack()

        text_area = scrolledtext.ScrolledText(
            self.main_area,
            font=("Courier", 11),
            bg="#2d2d2d",
            fg="white",
            insertbackground="white",
            wrap=tk.WORD,
            padx=20, pady=20
        )
        text_area.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        text_area.insert(tk.END, text)
        text_area.config(state=tk.DISABLED)

    # ---------------- Funciones de análisis ----------------
    def show_top_winners(self):
        """Muestra los pilotos con más victorias"""
        if self.results.empty or self.drivers.empty:
            messagebox.showwarning("Datos", "No hay datos suficientes para mostrar Top Victorias.")
            return

        wins = self.results[self.results['positionOrder'] == 1].copy()
        driver_wins = wins.groupby('driverId').size().reset_index(name='victories')
        driver_wins = driver_wins.merge(self.drivers[['driverId', 'forename', 'surname']], on='driverId', how='left')
        driver_wins['driver_name'] = driver_wins['forename'].fillna('') + ' ' + driver_wins['surname'].fillna('')
        top_10 = driver_wins.nlargest(10, 'victories')[['driver_name', 'victories']]

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.barh(top_10['driver_name'], top_10['victories'], color='#e10600')
        ax.set_xlabel('Número de Victorias', fontsize=12)
        ax.set_title('Top 10 Pilotos con Más Victorias', fontsize=16, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        self.show_plot_in_main_area(fig, "Top 10 Pilotos - Más Victorias")

    def show_top_podiums(self):
        """Muestra los pilotos con más podios"""
        if self.results.empty or self.drivers.empty:
            messagebox.showwarning("Datos", "No hay datos suficientes para mostrar Top Podios.")
            return

        podiums = self.results[self.results['positionOrder'].isin([1, 2, 3])].copy()
        driver_podiums = podiums.groupby('driverId').size().reset_index(name='podiums')
        driver_podiums = driver_podiums.merge(self.drivers[['driverId', 'forename', 'surname']], on='driverId', how='left')
        driver_podiums['driver_name'] = driver_podiums['forename'].fillna('') + ' ' + driver_podiums['surname'].fillna('')
        top_10 = driver_podiums.nlargest(10, 'podiums')[['driver_name', 'podiums']]

        fig, ax = plt.subplots(figsize=(12, 7))
        colors = ['#FFD700', '#C0C0C0', '#CD7F32'] * 4
        ax.barh(top_10['driver_name'], top_10['podiums'], color=colors[:len(top_10)])
        ax.set_xlabel('Número de Podios', fontsize=12)
        ax.set_title('Top 10 Pilotos con Más Podios', fontsize=16, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        self.show_plot_in_main_area(fig, "Top 10 Pilotos - Más Podios")

    def show_top_points(self):
        """Muestra los pilotos con más puntos"""
        if self.results.empty or self.drivers.empty:
            messagebox.showwarning("Datos", "No hay datos suficientes para mostrar Top Puntos.")
            return

        driver_points = self.results.groupby('driverId')['points'].sum().reset_index()
        driver_points = driver_points.merge(self.drivers[['driverId', 'forename', 'surname']], on='driverId', how='left')
        driver_points['driver_name'] = driver_points['forename'].fillna('') + ' ' + driver_points['surname'].fillna('')
        top_10 = driver_points.nlargest(10, 'points')[['driver_name', 'points']]

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.barh(top_10['driver_name'], top_10['points'], color='#00D2BE')
        ax.set_xlabel('Puntos Totales', fontsize=12)
        ax.set_title('Top 10 Pilotos con Más Puntos', fontsize=16, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        self.show_plot_in_main_area(fig, "Top 10 Pilotos - Más Puntos")

    def search_driver(self):
        """Busca un piloto específico"""
        if self.drivers.empty or self.results.empty:
            messagebox.showwarning("Datos", "No hay datos cargados para buscar pilotos.")
            return

        search_window = tk.Toplevel(self.root)
        search_window.title("Buscar Piloto")
        search_window.geometry("450x180")
        search_window.configure(bg="#2d2d2d")

        tk.Label(search_window, text="Ingresa el nombre o apellido del piloto:", font=("Arial", 11), bg="#2d2d2d",
                 fg="white").pack(pady=10)

        entry = tk.Entry(search_window, font=("Arial", 12), width=35)
        entry.pack(pady=5)
        entry.focus()

        def do_search():
            search_term = entry.get().strip()
            if not search_term:
                messagebox.showwarning("Advertencia", "Debes ingresar un nombre")
                return

            matches = self.drivers[
                self.drivers['forename'].str.contains(search_term, case=False, na=False) |
                self.drivers['surname'].str.contains(search_term, case=False, na=False)
            ]

            if matches.empty:
                messagebox.showinfo("No encontrado", f"No se encontró ningún piloto con '{search_term}'")
                return

            driver = matches.iloc[0]
            driver_id = driver['driverId']
            driver_results = self.results[self.results['driverId'] == driver_id]

            result_text = f"""
{'=' * 60}
  {driver.get('forename', '')} {driver.get('surname', '')}
{'=' * 60}

Nacionalidad: {driver.get('nationality', 'N/A')}
Fecha de nacimiento: {driver.get('dob', 'N/A')}

ESTADÍSTICAS:
  • Carreras: {len(driver_results)}
  • Victorias: {len(driver_results[driver_results['positionOrder'] == 1])}
  • Podios: {len(driver_results[driver_results['positionOrder'].isin([1, 2, 3])])}
  • Puntos totales: {driver_results['points'].sum():.1f}
  • Posición promedio: {driver_results['positionOrder'].mean():.2f if not driver_results.empty else 'N/A'}
"""
            search_window.destroy()
            self.show_text_result(result_text, f"🔍 Información de {driver.get('forename','')} {driver.get('surname','')}")

        tk.Button(search_window, text="Buscar", command=do_search, font=("Arial", 11), bg="#e10600", fg="white",
                  cursor="hand2").pack(pady=10)
        entry.bind("<Return>", lambda e: do_search())

    def show_top_constructor_wins(self):
        """Muestra los equipos con más victorias"""
        if self.results.empty or self.constructors.empty:
            messagebox.showwarning("Datos", "No hay datos suficientes para mostrar Top Equipos.")
            return

        wins = self.results[self.results['positionOrder'] == 1]
        constructor_wins = wins.groupby('constructorId').size().reset_index(name='victories')
        constructor_wins = constructor_wins.merge(self.constructors[['constructorId', 'name']], on='constructorId',
                                                  how='left')
        top_10 = constructor_wins.nlargest(10, 'victories')[['name', 'victories']]

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.barh(top_10['name'], top_10['victories'], color='#1e3a8a')
        ax.set_xlabel('Número de Victorias', fontsize=12)
        ax.set_title('Top 10 Equipos con Más Victorias', fontsize=16, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        self.show_plot_in_main_area(fig, "Top 10 Equipos - Más Victorias")

    def show_top_constructor_points(self):
        """Muestra los equipos con más puntos"""
        if self.results.empty or self.constructors.empty:
            messagebox.showwarning("Datos", "No hay datos suficientes para mostrar puntos por constructor.")
            return

        constructor_points = self.results.groupby('constructorId')['points'].sum().reset_index()
        constructor_points = constructor_points.merge(self.constructors[['constructorId', 'name']], on='constructorId',
                                                      how='left')
        top_10 = constructor_points.nlargest(10, 'points')[['name', 'points']]

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.barh(top_10['name'], top_10['points'], color='#16a34a')
        ax.set_xlabel('Puntos Totales', fontsize=12)
        ax.set_title('Top 10 Equipos con Más Puntos', fontsize=16, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        self.show_plot_in_main_area(fig, "Top 10 Equipos - Más Puntos")

    def show_constructor_evolution(self):
        """Muestra la evolución de un equipo"""
        if self.results.empty or self.constructors.empty or self.races.empty:
            messagebox.showwarning("Datos", "No hay datos suficientes para mostrar evolución de equipo.")
            return

        search_window = tk.Toplevel(self.root)
        search_window.title("Evolución de Equipo")
        search_window.geometry("450x200")
        search_window.configure(bg="#2d2d2d")

        tk.Label(search_window, text="Ingresa el nombre del equipo:", font=("Arial", 11), bg="#2d2d2d", fg="white").pack(
            pady=10)
        entry = tk.Entry(search_window, font=("Arial", 12), width=35)
        entry.pack(pady=5)
        entry.focus()

        def do_search():
            search_term = entry.get().strip()
            if not search_term:
                messagebox.showwarning("Advertencia", "Debes ingresar un nombre")
                return

            matches = self.constructors[self.constructors['name'].str.contains(search_term, case=False, na=False)]
            if matches.empty:
                messagebox.showinfo("No encontrado", f"No se encontró ningún equipo con '{search_term}'")
                return

            constructor = matches.iloc[0]
            const_results = self.results[self.results['constructorId'] == constructor['constructorId']]
            const_results = const_results.merge(self.races[['raceId', 'year']], on='raceId', how='left')

            if const_results.empty:
                messagebox.showinfo("Sin datos", f"No hay datos históricos para {constructor['name']}")
                return

            yearly_stats = const_results.groupby('year').agg({
                'points': 'sum',
                'positionOrder': lambda x: (x == 1).sum()
            }).reset_index()
            yearly_stats.columns = ['year', 'points', 'wins']

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            ax1.plot(yearly_stats['year'], yearly_stats['points'], marker='o', linewidth=2, color='#e10600')
            ax1.set_xlabel('Año', fontsize=11)
            ax1.set_ylabel('Puntos', fontsize=11)
            ax1.set_title(f'Evolución de Puntos - {constructor["name"]}', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)

            ax2.bar(yearly_stats['year'], yearly_stats['wins'], color='#00D2BE', alpha=0.7)
            ax2.set_xlabel('Año', fontsize=11)
            ax2.set_ylabel('Victorias', fontsize=11)
            ax2.set_title(f'Victorias por Año - {constructor["name"]}', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            search_window.destroy()
            self.show_plot_in_main_area(fig, f"📈 Evolución de {constructor['name']}")

        tk.Button(search_window, text="Analizar", command=do_search, font=("Arial", 11), bg="#e10600", fg="white",
                  cursor="hand2").pack(pady=12)
        entry.bind("<Return>", lambda e: do_search())

    def compare_drivers(self):
        """Compara dos pilotos"""
        if self.results.empty or self.drivers.empty:
            messagebox.showwarning("Datos", "No hay datos suficientes para comparar pilotos.")
            return

        compare_window = tk.Toplevel(self.root)
        compare_window.title("Comparar Pilotos")
        compare_window.geometry("450x300")
        compare_window.configure(bg="#2d2d2d")

        tk.Label(compare_window, text="Comparación de Pilotos", font=("Arial", 14, "bold"), bg="#2d2d2d",
                 fg="white").pack(pady=10)
        tk.Label(compare_window, text="Primer piloto (apellido):", font=("Arial", 10), bg="#2d2d2d",
                 fg="white").pack(pady=5)
        entry1 = tk.Entry(compare_window, font=("Arial", 11), width=35)
        entry1.pack(pady=5)
        tk.Label(compare_window, text="Segundo piloto (apellido):", font=("Arial", 10), bg="#2d2d2d",
                 fg="white").pack(pady=5)
        entry2 = tk.Entry(compare_window, font=("Arial", 11), width=35)
        entry2.pack(pady=5)

        def do_compare():
            driver1_name = entry1.get().strip()
            driver2_name = entry2.get().strip()

            if not driver1_name or not driver2_name:
                messagebox.showwarning("Advertencia", "Debes ingresar ambos apellidos")
                return

            driver1 = self.drivers[
                self.drivers['forename'].str.contains(driver1_name, case=False, na=False) |
                self.drivers['surname'].str.contains(driver1_name, case=False, na=False)
            ]
            driver2 = self.drivers[
                self.drivers['forename'].str.contains(driver2_name, case=False, na=False) |
                self.drivers['surname'].str.contains(driver2_name, case=False, na=False)
            ]

            if driver1.empty or driver2.empty:
                messagebox.showerror("Error", "No se encontraron uno o ambos pilotos")
                return

            d1 = driver1.iloc[0]
            d2 = driver2.iloc[0]

            d1_results = self.results[self.results['driverId'] == d1['driverId']]
            d2_results = self.results[self.results['driverId'] == d2['driverId']]

            d1_name = f"{d1.get('forename','')} {d1.get('surname','')}"
            d2_name = f"{d2.get('forename','')} {d2.get('surname','')}"

            stats = {
                'Carreras': [len(d1_results), len(d2_results)],
                'Victorias': [
                    len(d1_results[d1_results['positionOrder'] == 1]),
                    len(d2_results[d2_results['positionOrder'] == 1])
                ],
                'Podios': [
                    len(d1_results[d1_results['positionOrder'].isin([1, 2, 3])]),
                    len(d2_results[d2_results['positionOrder'].isin([1, 2, 3])])
                ],
                'Puntos': [
                    d1_results['points'].sum() if not d1_results.empty else 0,
                    d2_results['points'].sum() if not d2_results.empty else 0
                ]
            }

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            colors = ['#e10600', '#00D2BE']
            metrics = ['Carreras', 'Victorias', 'Podios', 'Puntos']

            for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
                values = stats[metric]
                ax.bar([d1_name, d2_name], values, color=colors)
                ax.set_ylabel(metric, fontsize=11)
                ax.set_title(metric, fontsize=12, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)

            plt.suptitle(f'Comparación: {d1_name} vs {d2_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()

            compare_window.destroy()
            self.show_plot_in_main_area(fig, f"⚔️ {d1_name} vs {d2_name}")

        tk.Button(compare_window, text="Comparar", command=do_compare, font=("Arial", 11), bg="#e10600", fg="white",
                  cursor="hand2", width=20).pack(pady=12)

    # ---------------- Predicción ML ----------------
    def show_model_performance(self):
        """Muestra el rendimiento del modelo de predicción"""
        if not self.model:
            messagebox.showerror("Error", "El modelo de predicción no está disponible")
            return

        fig = plt.figure(figsize=(14, 10))

        # Real vs Predicho
        ax1 = plt.subplot(2, 2, 1)
        sns.scatterplot(x=self.y_test, y=self.y_pred, alpha=0.6, ax=ax1)
        ax1.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()],
                 'r--', lw=2, label='Predicción perfecta')
        ax1.set_xlabel("Puntos Reales", fontsize=11)
        ax1.set_ylabel("Puntos Predichos", fontsize=11)
        ax1.set_title("Predicción vs Realidad", fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Distribución de errores
        ax2 = plt.subplot(2, 2, 2)
        errors = self.y_test - self.y_pred
        ax2.hist(errors, bins=50, edgecolor='black')
        ax2.set_xlabel("Error de Predicción", fontsize=11)
        ax2.set_ylabel("Frecuencia", fontsize=11)
        ax2.set_title("Distribución de Errores", fontsize=12, fontweight='bold')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.grid(True, alpha=0.3)

        # Métricas
        ax3 = plt.subplot(2, 2, 3)
        ax3.axis('off')
        metrics_text = f"""
MÉTRICAS DEL MODELO
{'=' * 40}

Algoritmo: Regresión Lineal

Error Cuadrático Medio (MSE):
{self.mse:.2f}

Coeficiente de Determinación (R²):
{self.r2:.3f}

Precisión del modelo:
{self.r2 * 100:.1f}%

{'=' * 40}
"""
        ax3.text(0.05, 0.5, metrics_text, fontsize=11, family='monospace', verticalalignment='center')

        # Primeras 10 predicciones
        ax4 = plt.subplot(2, 2, 4)
        comparison_df = pd.DataFrame({'Real': self.y_test.values[:10], 'Predicho': self.y_pred[:10]})
        x_pos = np.arange(len(comparison_df))
        width = 0.35
        ax4.bar(x_pos - width / 2, comparison_df['Real'], width, label='Real')
        ax4.bar(x_pos + width / 2, comparison_df['Predicho'], width, label='Predicho')
        ax4.set_xlabel("Muestra", fontsize=11)
        ax4.set_ylabel("Puntos", fontsize=11)
        ax4.set_title("Primeras 10 Predicciones", fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        self.show_plot_in_main_area(fig, f"🔮 Rendimiento del Modelo ML - Precisión: {self.r2 * 100:.1f}%")

    def predict_driver_points(self):
        """Predice los puntos de un piloto específico"""
        if not self.model:
            messagebox.showwarning("Advertencia", "El modelo de predicción no está disponible")
            return

        predict_window = tk.Toplevel(self.root)
        predict_window.title("Predecir Puntos de Piloto")
        predict_window.geometry("480x220")
        predict_window.configure(bg="#2d2d2d")

        tk.Label(predict_window, text="🔮 Predicción de Puntos con Machine Learning", font=("Arial", 13, "bold"),
                 bg="#2d2d2d", fg="#9333ea").pack(pady=10)
        tk.Label(predict_window, text="Ingresa el nombre o apellido del piloto:", font=("Arial", 10), bg="#2d2d2d",
                 fg="white").pack(pady=5)
        entry = tk.Entry(predict_window, font=("Arial", 12), width=40)
        entry.pack(pady=5)
        entry.focus()

        def do_predict():
            search_term = entry.get().strip()
            if not search_term:
                messagebox.showwarning("Advertencia", "Debes ingresar un nombre")
                return

            matches = self.drivers[
                self.drivers['forename'].str.contains(search_term, case=False, na=False) |
                self.drivers['surname'].str.contains(search_term, case=False, na=False)
            ]

            if matches.empty:
                messagebox.showinfo("No encontrado", f"No se encontró ningún piloto con '{search_term}'")
                return

            driver = matches.iloc[0]
            driver_id = driver['driverId']
            driver_name = f"{driver.get('forename','')} {driver.get('surname','')}"

            # Preparar datos del piloto a partir de df_clean si existe
            if self.df_clean is None or self.df_clean.empty:
                messagebox.showinfo("Sin datos", "No hay dataset limpio para obtener historiales del piloto.")
                return

            data = self.df_clean.replace('\\N', pd.NA)
            cols = ['points', 'position', 'grid', 'laps', 'constructorId', 'circuitId', 'driverId']
            if not all(c in data.columns for c in cols):
                messagebox.showinfo("Sin columnas", "El dataset limpio no contiene las columnas requeridas para predecir.")
                return

            piloto_datos = data[data['driverId'] == driver_id][cols].dropna()
            if piloto_datos.empty:
                messagebox.showinfo("Sin datos", f"No hay datos suficientes para {driver_name}")
                return

            X_piloto = piloto_datos[['position', 'grid', 'laps', 'constructorId', 'circuitId', 'driverId']].astype(object)
            X_piloto_encoded = pd.get_dummies(X_piloto, columns=['constructorId', 'circuitId', 'driverId'], drop_first=True)
            X_piloto_encoded = X_piloto_encoded.reindex(columns=self.X_train.columns, fill_value=0)

            predicciones = self.model.predict(X_piloto_encoded)

            # Visualización simple comparativa
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            indices = range(min(len(piloto_datos), 50))
            ax1.plot(indices, piloto_datos['points'].values[:50], label='Puntos Reales', marker='o', linewidth=2)
            ax1.plot(indices, predicciones[:50], label='Puntos Predichos', marker='s', linewidth=2)
            ax1.set_xlabel('Registro', fontsize=11)
            ax1.set_ylabel('Puntos', fontsize=11)
            ax1.set_title(f'Predicción vs Realidad - {driver_name}', fontsize=13, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.scatter(piloto_datos['points'].values, predicciones, alpha=0.6, s=80)
            minv = min(piloto_datos['points'].min(), predicciones.min())
            maxv = max(piloto_datos['points'].max(), predicciones.max())
            ax2.plot([minv, maxv], [minv, maxv], 'r--', lw=2)
            ax2.set_xlabel('Puntos Reales', fontsize=11)
            ax2.set_ylabel('Puntos Predichos', fontsize=11)
            ax2.set_title('Correlación de Predicciones', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            error_promedio = np.mean(np.abs(piloto_datos['points'].values - predicciones))

            predict_window.destroy()
            self.show_plot_in_main_area(fig, f"Predicción para {driver_name} - Error promedio: {error_promedio:.2f} pts")

        tk.Button(predict_window, text="Predecir", command=do_predict, font=("Arial", 11), bg="#9333ea", fg="white",
                  cursor="hand2", width=20).pack(pady=12)
        entry.bind("<Return>", lambda e: do_predict())

    # ---------------- Visualizaciones avanzadas ----------------
    def show_heatmap(self):
        """Muestra mapa de calor de victorias por década"""
        if self.results.empty or self.races.empty or self.drivers.empty:
            messagebox.showwarning("Datos", "No hay datos suficientes para el mapa de calor.")
            return

        wins = self.results[self.results['positionOrder'] == 1].copy()
        wins = wins.merge(self.races[['raceId', 'year']], on='raceId', how='left')
        wins = wins.merge(self.drivers[['driverId', 'forename', 'surname']], on='driverId', how='left')
        wins['driver_name'] = wins['forename'].fillna('') + ' ' + wins['surname'].fillna('')
        wins['decade'] = (wins['year'] // 10) * 10

        top_drivers = wins['driver_name'].value_counts().head(10).index
        decade_wins = wins[wins['driver_name'].isin(top_drivers)].groupby(['driver_name', 'decade']).size().reset_index(
            name='wins')

        pivot_table = decade_wins.pivot(index='driver_name', columns='decade', values='wins').fillna(0)

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlOrRd', cbar_kws={'label': 'Victorias'}, ax=ax)
        ax.set_title('Victorias por Década - Top 10 Pilotos', fontsize=16, fontweight='bold')
        ax.set_xlabel('Década', fontsize=12)
        ax.set_ylabel('Piloto', fontsize=12)
        plt.tight_layout()

        self.show_plot_in_main_area(fig, "Mapa de Calor - Victorias por Década")

    def show_distribution(self):
        """Muestra la distribución de posiciones finales"""
        if self.results.empty:
            messagebox.showwarning("Datos", "No hay datos suficientes para la distribución.")
            return

        fig, ax = plt.subplots(figsize=(14, 7))
        position_counts = self.results['positionOrder'].value_counts().sort_index().head(20)
        ax.bar(position_counts.index.astype(str), position_counts.values, alpha=0.8)
        ax.set_xlabel('Posición Final', fontsize=12)
        ax.set_ylabel('Frecuencia', fontsize=12)
        ax.set_title('Distribución de Posiciones Finales en F1', fontsize=16, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        self.show_plot_in_main_area(fig, "Distribución de Posiciones Finales")

    def show_top5_evolution(self):
        """Muestra la evolución de los top 5 pilotos"""
        if self.results.empty or self.drivers.empty or self.races.empty:
            messagebox.showwarning("Datos", "No hay datos suficientes para mostrar evolución Top 5.")
            return

        top5_drivers = self.results.groupby('driverId')['points'].sum().nlargest(5)

        fig, ax = plt.subplots(figsize=(16, 8))

        for driver_id in top5_drivers.index:
            driver_info = self.drivers[self.drivers['driverId'] == driver_id]
            if driver_info.empty:
                continue
            driver_info = driver_info.iloc[0]
            driver_name = f"{driver_info.get('forename','')} {driver_info.get('surname','')}"
            driver_results = self.results[self.results['driverId'] == driver_id].copy()
            driver_results = driver_results.merge(self.races[['raceId', 'year']], on='raceId', how='left')
            if driver_results.empty:
                continue
            yearly_points = driver_results.groupby('year')['points'].sum().reset_index()
            ax.plot(yearly_points['year'], yearly_points['points'], marker='o', linewidth=2, label=driver_name)

        ax.set_xlabel('Año', fontsize=12)
        ax.set_ylabel('Puntos', fontsize=12)
        ax.set_title('Evolución de Puntos - Top 5 Pilotos Históricos', fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        self.show_plot_in_main_area(fig, "Evolución Top 5 Pilotos")

    # ---------------- Estadísticas generales ----------------
    def show_general_stats(self):
        """Muestra estadísticas generales"""
        if self.drivers.empty or self.races.empty or self.results.empty:
            messagebox.showwarning("Datos", "No hay datos suficientes para mostrar estadísticas generales.")
            return

        top_nat_name = self.drivers['nationality'].value_counts().index[0] if not self.drivers['nationality'].dropna().empty else 'N/A'
        top_nationality = self.drivers['nationality'].value_counts().iloc[0] if not self.drivers['nationality'].dropna().empty else 0

        model_stats = ""
        if self.model:
            model_stats = f"""
🔮 MODELO DE MACHINE LEARNING:
  • Algoritmo: Regresión Lineal
  • Precisión (R²): {self.r2 * 100:.1f}%
  • Error cuadrático medio: {self.mse:.2f}
"""

        stats_text = f"""
{'=' * 80}
  ESTADÍSTICAS GENERALES DEL DATASET DE FÓRMULA 1
{'=' * 80}

📊 INFORMACIÓN GENERAL:
  • Total de pilotos registrados: {len(self.drivers)}
  • Total de carreras históricas: {len(self.races)}
  • Total de equipos/constructores: {len(self.constructors)}
  • Período cubierto: {self.races['year'].min() if not self.races.empty else 'N/A'} - {self.races['year'].max() if not self.races.empty else 'N/A'}
  • Total de resultados registrados: {len(self.results)}
  • Total de vueltas registradas: {len(self.lap_times)}

🏁 DATOS DE CARRERAS:
  • Número de temporadas: {len(self.races['year'].unique()) if not self.races.empty else 0}
  • Promedio de carreras por temporada: {(len(self.races) / len(self.races['year'].unique())):.1f if not self.races.empty else 0}
  • Circuitos únicos: {len(self.races['circuitId'].unique()) if 'circuitId' in self.races.columns else 'N/A'}

👨‍✈️ DATOS DE PILOTOS:
  • Nacionalidad más común: {top_nat_name} ({top_nationality} pilotos)
  • Pilotos con al menos 1 victoria: {len(self.results[self.results['positionOrder'] == 1]['driverId'].unique())}
  • Pilotos con al menos 1 podio: {len(self.results[self.results['positionOrder'] <= 3]['driverId'].unique())}

{model_stats}
{'=' * 80}
"""
        self.show_text_result(stats_text, "Estadísticas Generales del Dataset")


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = F1AnalyzerGUI(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error Fatal", f"Error al iniciar la aplicación:\n{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
