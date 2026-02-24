# 1. Importamos las herramientas necesarias
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# 2. Datos de entrenamiento (Dataset)
# En un proyecto real, esto vendría de un archivo CSV o base de datos.
textos = [
    "Me encanta este producto, es increíble", 
    "Excelente calidad y muy rápido",
    "El mejor software que he usado",
    "No me gusta, es muy lento",
    "Pésimo servicio y mala calidad",
    "Es lo peor, no lo compren",
    "Funciona muy bien y es barato",
    "Horrible experiencia, estoy decepcionado"
]

# Etiquetas: 1 para Positivo, 0 para Negativo
etiquetas = [1, 1, 1, 0, 0, 0, 1, 0]

# 3. Creación del Modelo
# Usamos un 'Pipeline' para combinar la conversión de texto a números 
# y el algoritmo de clasificación en un solo paso.
modelo = make_pipeline(CountVectorizer(), MultinomialNB())

# 4. Entrenamiento
# Aquí el modelo aprende la relación entre las palabras y las etiquetas.
modelo.fit(textos, etiquetas)

# 5. Función para probar el modelo
def analizar_sentimiento(frase):
    prediccion = modelo.predict([frase])
    sentimiento = "Positivo" if prediccion[0] == 1 else "Negativo"
    return sentimiento

# --- PRUEBA DEL CÓDIGO ---
nueva_frase = "Este código es fantástico y funciona perfecto"
resultado = analizar_sentimiento(nueva_frase)

print(f"Frase: '{nueva_frase}'")
print(f"Resultado del análisis: {resultado}")