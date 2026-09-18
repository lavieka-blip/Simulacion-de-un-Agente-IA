import json
import ollama
import chromadb

MODELO_EMBED = "nomic-embed-text"   # Convierte texto -> vector.
MODELO_GEN   = "gemma3:4b"   

# ==========================================
# DATOS DUMMY (Simulación de RAG)
# ==========================================

GUIA_TURISTICA = [
    "La Pinta: Barco histórico en Baiona. Precio: 4€. Tags: histórico, cultura.",
    "Rocamar: Restaurante de marisco en Baredo. Precio: 40€. Tags: marisco, opciónes veganas y vegetarianas.",
    "Virgen de la Roca: Parque natural en Baiona. Precio: Gratis. Tags: parque, naturaleza.",
    "Los abetos: Restaurante asador en Nigrán. Precio: 10€. Tags: asador, sin opciones vegetarianas."
]

# ---------------------------------------------------------------------------
# 1) FUNCIÓN DE EMBEDDING: texto -> vector de números
# ---------------------------------------------------------------------------
def obtener_embedding(texto):
    # Ollama devuelve el vector que representa el "significado" del texto.
    respuesta = ollama.embeddings(model=MODELO_EMBED, prompt=texto)
    return respuesta["embedding"]

# ---------------------------------------------------------------------------
# 2) INDEXAR: crear la base vectorial y meter los sitios turisticos y restaurantes
# ---------------------------------------------------------------------------
def crear_indice():
    cliente = chromadb.Client()

    # La colección es como una "tabla" de vectores. Usamos distancia de coseno.
    coleccion = cliente.create_collection(
        name="guia_turistica",
        metadata={"hnsw:space": "cosine"},
    )

    # Vectorizamos cada apunte y lo guardamos con un id y su texto original.
    for i, texto in enumerate(GUIA_TURISTICA):
        coleccion.add(
            ids=[f"location_{i}"],
            documents=[texto],
            embeddings=[obtener_embedding(texto)],
        )

    print(f"[OK] Indexados {len(GUIA_TURISTICA)} apuntes en ChromaDB.\n")
    return coleccion

# ==========================================
# VERSIÓN A: Solo Prompt y Contexto
# ==========================================
def agent_version_a(pregunta):
    system_prompt = (
        "Eres un agente de viajes experto. Responde siempre con un tono profesional, "
        "en formato de lista estructurada por momentos del día (Mañana, Almuerzo, Tarde)."
    )
    # Al no tener RAG, el modelo depende solo de su conocimiento previo implícito
    prompt_completo = f"System: {system_prompt}\nUser: {pregunta}"
    
    # SIMULACIÓN DE RESPUESTA DEL LLM
    respuesta = ollama.chat(
        model=MODELO_GEN,
        messages=[{"role": "user", "content": pregunta}],
        options={"temperature": 0.2,  "top_k": 200},
    )
    print(f"[SIN RAG] {pregunta}")
    print(f"  RESPUESTA: {respuesta['message']['content']}\n")

# ==========================================
# VERSIÓN B: Prompt + RAG (Datos Simulados)
# ==========================================

def agent_version_b(coleccion, pregunta, n_resultados=1):

    
    resultado = coleccion.query(
        query_embeddings=[obtener_embedding(pregunta)],
        n_results=n_resultados,
    )
    fragmentos = resultado["documents"][0]      # textos recuperados
    distancias = resultado["distances"][0]      # menor = más parecido

    # Mostramos qué recuperó (para ver el "retrieval" en acción en clase):
    print(f"PREGUNTA: {pregunta}")
    for frag, dist in zip(fragmentos, distancias):
        print(f"  -> Recuperado (distancia {dist:.3f}): {frag}")

    # --- GENERAR: metemos el contexto recuperado en el prompt ---
    contexto = "\n".join(fragmentos)

    prompt = (
        "Eres un agente de viajes experto. "
        f"INFORMACIÓN:\n{contexto}\n\n"
        f"PREGUNTA: {pregunta}"
    )

    respuesta = ollama.chat(
        model=MODELO_GEN,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3,  "top_k": 200},   # 0 = se ciñe al contexto, no improvisa
    )
    print(f"  RESPUESTA: {respuesta['message']['content']}\n")

# ==========================================
# VERSIÓN C: Prompt + RAG + Parámetros
# ==========================================
def agent_version_c(coleccion, pregunta, n_resultados=1):

    
    resultado = coleccion.query(
        query_embeddings=[obtener_embedding(pregunta)],
        n_results=n_resultados,
    )
    fragmentos = resultado["documents"][0]      # textos recuperados
    distancias = resultado["distances"][0]      # menor = más parecido

    # Mostramos qué recuperó (para ver el "retrieval" en acción en clase):
    print(f"PREGUNTA: {pregunta}")
    for frag, dist in zip(fragmentos, distancias):
        print(f"  -> Recuperado (distancia {dist:.3f}): {frag}")

    # --- GENERAR: metemos el contexto recuperado en el prompt ---
    contexto = "\n".join(fragmentos)

    prompt = (
        "Eres un agente de viajes de lujo. "
        "Tu tono es elegante pero cercano. "
        "Requisito estricto: Formatea la respuesta estrictamente en una tabla Markdown con las columnas: "
        "[Horario | Actividad | Restricción Dietética / Notas | Precio]."
        f"INFORMACIÓN:\n{contexto}\n\n"
        f"PREGUNTA: {pregunta}"
    )

    respuesta = ollama.chat(
        model=MODELO_GEN,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.7,  "top_k": 200},   # 0 = se ciñe al contexto, no improvisa
    )
    print(f"  RESPUESTA: {respuesta['message']['content']}\n")

# ==========================================
# EJECUCIÓN DE LA SIMULACIÓN
# ==========================================
if __name__ == "__main__":
    print("=== DEMO AGENTE A: SIN RAG (Solo prompt y contexto. Temperatura 0.2) ===\n")
    agent_version_a("Planifícame un día en Baiona. Soy vegetariano y busco un ritmo relajado.")

    print("=== DEMO AGENTE B: CON RAG (prompt + contexto + RAG. Temperatura 0.3) ===\n")
    coleccion = crear_indice()
    agent_version_b(coleccion, "Planifícame un día en Baiona. Soy vegetariano y busco un ritmo relajado.")

    print("=== DEMO AGENTE C: CON RAG (prompt más extenso + contexto + RAG. Temperatura 0.7) ===\n")
    agent_version_c(coleccion, "Planifícame un día en Baiona, Baredo y Nigrán.")
