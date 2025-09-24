#!/usr/bin/env bash
# Crea un entorno virtual, lo activa, instala dependencias y ejecuta el script.

# 1. Crear entorno virtual (carpeta: venv_cartpole)
python3 -m venv venv_cartpole

# 2. Activar entorno
source venv_cartpole/bin/activate

# 3. Actualizar pip y setuptools
pip install --upgrade pip setuptools

pip install "gymnasium[classic_control]"

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Ejecutar tu script principal
python testing.py
