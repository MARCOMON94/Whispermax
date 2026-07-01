# Whispermax

Aplicacion local con FastAPI para subir un video, extraer su audio con `ffmpeg`, transcribirlo con `whisper` y guardar la transcripcion en una subcarpeta.

## Requisitos

- Python 3.10-3.12 recomendado para Whisper/Torch. Python 3.14 puede funcionar si hay wheels compatibles disponibles.
- `ffmpeg` instalado y disponible en el `PATH`.

## Instalacion

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Uso

```powershell
python main.py
```

Se abrira una ventana del navegador en `http://127.0.0.1:8000`. Sube uno o varios videos, elige el modelo de Whisper y pulsa `Anadir a cola`. El modelo por defecto es `tiny`, que es el mas rapido.

Cuando subes varios archivos, la app los deja en cola y los transcribe de uno en uno. La pagina de estado se actualiza sola y muestra los enlaces de descarga cuando cada archivo termina.

La cola incluye:

- Boton `Cancelar` independiente para cada video.
- Barra de progreso aproximada por fases.
- Limpieza automatica del video subido y del audio WAV temporal al completar, fallar o cancelar.
- Selector de consumo: `Bajo` usa menos hilos y prioridad baja para evitar picos de CPU.

## Limites de consumo

Por defecto:

- Se procesa solo 1 video a la vez.
- `Bajo`: 1 hilo para Whisper/Torch y 1 hilo para `ffmpeg`.
- Maximo 55 videos por tanda.
- Maximo 2048 MB por archivo subido.

Puedes cambiar los limites antes de arrancar:

```powershell
$env:WHISPERMAX_MAX_BATCH_FILES = "3"
$env:WHISPERMAX_MAX_UPLOAD_MB = "1024"
python main.py
```

Los resultados quedan en:

- `salidas/videos`: videos subidos.
- `salidas/audio`: audio extraido en WAV.
- `salidas/modelos`: modelos descargados por Whisper.
- `salidas/transcripciones`: transcripcion en `.docx` y `.txt`.

Tambien puedes iniciar el servidor sin abrir el navegador:

```powershell
python main.py --no-browser
```

Y puedes cambiar el puerto si el `8000` ya estuviera ocupado:

```powershell
python main.py --port 8010
```
