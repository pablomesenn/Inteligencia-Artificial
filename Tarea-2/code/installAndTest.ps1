# Crea un entorno virtual, lo activa, instala dependencias y ejecuta el script.

# 1. Crear entorno virtual (carpeta: venv_cartpole)
python -m venv venv_cartpole

# 2. Activar entorno virtual
& venv_cartpole\Scripts\Activate.ps1

# 3. Actualizar pip y setuptools
python -m pip install --upgrade pip setuptools

# 4. Instalar gymnasium con classic_control
python -m pip install "gymnasium[classic_control]"

# 5. Instalar dependencias desde requirements.txt
python -m pip install -r requirements.txt

# 6. Ejecutar tu script principal
python testing.py
