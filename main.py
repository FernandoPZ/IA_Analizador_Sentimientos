import customtkinter as ctk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# --- PARTE 1: LÓGICA DE INTELIGENCIA ARTIFICIAL ---
textos = [
    "Me encanta este producto", "Excelente calidad", "El mejor software",
    "No me gusta, es muy lento", "Pésimo servicio", "Es lo peor",
    "Funciona muy bien", "Horrible experiencia", "Me siento feliz con la compra",
    "No funciona, dinero tirado"
]
etiquetas = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]

# Entrenamos el modelo
modelo = make_pipeline(CountVectorizer(), MultinomialNB())
modelo.fit(textos, etiquetas)

# --- PARTE 2: INTERFAZ GRÁFICA ---
class AppIA(ctk.CTk):
    def __init__(self):
        super().__init__()
        # Configuración de la ventana
        self.title("Analizador de Sentimientos IA")
        self.geometry("450x300")
        ctk.set_appearance_mode("dark") # Modo oscuro por defecto
        # Título
        self.label_titulo = ctk.CTkLabel(self, text="Detector de Sentimientos", font=("Arial", 20, "bold"))
        self.label_titulo.pack(pady=20)
        # Campo de entrada de texto
        self.entrada_texto = ctk.CTkEntry(self, placeholder_text="Escribe tu reseña aquí...", width=300)
        self.entrada_texto.pack(pady=10)
        # Botón de Análisis
        self.boton_analizar = ctk.CTkButton(self, text="Analizar Texto", command=self.ejecutar_analisis)
        self.boton_analizar.pack(pady=10)
        # Etiqueta de Resultado
        self.resultado_label = ctk.CTkLabel(self, text="Resultado: Esperando...", font=("Arial", 16))
        self.resultado_label.pack(pady=20)
    def ejecutar_analisis(self):
        frase = self.entrada_texto.get()
        if frase:
            prediccion = modelo.predict([frase])[0]
            if prediccion == 1:
                self.resultado_label.configure(text="Resultado: POSITIVO 😊", text_color="#2ecc71")
            else:
                self.resultado_label.configure(text="Resultado: NEGATIVO 😡", text_color="#e74c3c")
        else:
            self.resultado_label.configure(text="Por favor, escribe algo.", text_color="yellow")

# Iniciar la aplicación
if __name__ == "__main__":
    app = AppIA()
    app.mainloop()