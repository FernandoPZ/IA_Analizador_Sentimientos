import customtkinter as ctk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import csv
import os

# --- PARTE 1: LÓGICA DE IA Y MANEJO DE DATOS ---
# 1. Datos base iniciales
textos = [
    "Me encanta este producto", "Excelente calidad", "El mejor software",
    "No me gusta, es muy lento", "Pésimo servicio", "Es lo peor",
    "Funciona muy bien", "Horrible experiencia", "Me siento feliz con la compra",
    "No funciona, dinero tirado"
]
etiquetas = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]

# 2. Cargar datos históricos del usuario
archivo_csv = "dataset.csv"
if os.path.exists(archivo_csv):
    with open(archivo_csv, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for fila in reader:
            if len(fila) == 2:
                textos.append(fila[0])          # Agrega la frase
                etiquetas.append(int(fila[1]))  # Agrega la etiqueta (0 o 1)

# 3. Entrenamos el modelo con los datos base + los datos guardados
modelo = make_pipeline(CountVectorizer(), MultinomialNB())
modelo.fit(textos, etiquetas)

# --- PARTE 2: INTERFAZ GRÁFICA ---
class AppIA(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Analizador de Sentimientos IA")
        self.geometry("450x380")
        ctk.set_appearance_mode("dark")
        # Variables para guardar el estado actual
        self.frase_actual = ""
        self.prediccion_actual = -1
        # Elementos de la interfaz (UI)
        self.label_titulo = ctk.CTkLabel(self, text="Detector de Sentimientos", font=("Arial", 20, "bold"))
        self.label_titulo.pack(pady=20)
        self.entrada_texto = ctk.CTkEntry(self, placeholder_text="Escribe tu reseña aquí...", width=300)
        self.entrada_texto.pack(pady=10)
        self.boton_analizar = ctk.CTkButton(self, text="Analizar Texto", command=self.ejecutar_analisis)
        self.boton_analizar.pack(pady=10)
        self.resultado_label = ctk.CTkLabel(self, text="Resultado: Esperando...", font=("Arial", 16))
        self.resultado_label.pack(pady=10)
        # Contenedor oculto para los botones de Feedback
        self.frame_feedback = ctk.CTkFrame(self, fg_color="transparent")
        self.boton_correcto = ctk.CTkButton(self.frame_feedback, text="✔️ Correcto", width=100, fg_color="green", hover_color="darkgreen", command=lambda: self.guardar_feedback(True))
        self.boton_correcto.pack(side="left", padx=10)
        self.boton_incorrecto = ctk.CTkButton(self.frame_feedback, text="❌ Se equivocó", width=100, fg_color="red", hover_color="darkred", command=lambda: self.guardar_feedback(False))
        self.boton_incorrecto.pack(side="right", padx=10)
    def ejecutar_analisis(self):
        self.frase_actual = self.entrada_texto.get()
        if self.frase_actual:
            self.prediccion_actual = modelo.predict([self.frase_actual])[0]
            if self.prediccion_actual == 1:
                self.resultado_label.configure(text="Resultado: POSITIVO 😊", text_color="#2ecc71")
            else:
                self.resultado_label.configure(text="Resultado: NEGATIVO 😡", text_color="#e74c3c")
            # Mostrar los botones de feedback
            self.frame_feedback.pack(pady=10)
        else:
            self.resultado_label.configure(text="Por favor, escribe algo.", text_color="yellow")
            self.frame_feedback.pack_forget() # Ocultar botones si no hay texto
    def guardar_feedback(self, es_correcto):
        # Determinar la etiqueta final que debemos guardar
        etiqueta_final = self.prediccion_actual if es_correcto else (1 - self.prediccion_actual)
        # Guardar en el archivo CSV
        with open(archivo_csv, mode="a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self.frase_actual, etiqueta_final])
        # Agradecer al usuario y limpiar la interfaz
        self.resultado_label.configure(text="¡Gracias! Aprenderé de esto 🧠", text_color="white")
        self.entrada_texto.delete(0, 'end')
        self.frame_feedback.pack_forget() # Ocultar botones nuevamente

if __name__ == "__main__":
    app = AppIA()
    app.mainloop()