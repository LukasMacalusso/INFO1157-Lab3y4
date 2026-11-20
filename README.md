# Proyectos #3 y #4 — Sistemas Inteligentes (INFO1157)

## Estructura

```text
Final_project/
├── assets/          # Datasets (.npy) y PDF del enunciado
├── proyecto3/       # Tareas 1–4 + Proyecto3_Laboratorio.ipynb
├── proyecto4/       # Tareas 5–8 + Proyecto4_Laboratorio.ipynb
├── requirements.txt # Dependencias mínimas
└── Makefile         # Ejecución rápida
```

## Uso

```bash
make setup     # Crea .venv e instala dependencias (si no existe)
make p3        # Ejecuta Proyecto 3 y guarda figuras en proyecto3/output/
make p4        # Ejecuta Proyecto 4 y guarda figuras en proyecto4/output/
make all       # Ejecuta ambos proyectos
make notebook  # Abre Jupyter Notebook
make clean     # Limpia salidas y caché
```
