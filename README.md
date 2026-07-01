# Whispermax

Aplicacion local con FastAPI para subir videos, extraer el audio con `ffmpeg`, transcribir con Whisper y guardar cada transcripcion como `.docx` y `.txt`.

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

Se abrira una ventana del navegador en `http://127.0.0.1:8000`. Sube uno o varios videos, elige el modelo de Whisper, el idioma y el consumo, y pulsa `Anadir a cola`. El modelo por defecto es `tiny`, que es el mas rapido.

Cuando subes varios archivos, la app los deja en cola y los transcribe de uno en uno. La pagina de estado se actualiza sola y muestra los enlaces de descarga cuando cada archivo termina.

La cola incluye:

- Boton `Cancelar` independiente para cada video.
- Panel limpio con un unico trabajo activo y su barra de progreso.
- Lista separada de pendientes, sin barras falsas al 0%.
- Durante `Transcribiendo`, la barra usa el porcentaje que aparece en `Whisper esta procesando el audio (N%)`.
- Limpieza automatica del video subido y del audio WAV temporal al completar, fallar o cancelar.
- Selector de consumo: `Bajo` usa menos hilos y prioridad baja para evitar picos de CPU.

Los archivos finales se conservan en `salidas/transcripciones`. Los videos temporales de `salidas/videos` y los WAV de `salidas/audio` se borran automaticamente cuando cada trabajo termina, falla o se cancela.

## Limites de consumo

Por defecto:

- Se procesa solo 1 video a la vez.
- `Bajo`: 1 hilo para Whisper/Torch y 1 hilo para `ffmpeg`.
- `Medio`: 2 hilos para Whisper/Torch y 1 hilo para `ffmpeg`.
- `Rapido`: 4 hilos para Whisper/Torch y 2 hilos para `ffmpeg`.
- `Ultrarrapido`: hasta 8 hilos para Whisper/Torch y 2 hilos para `ffmpeg`.
- Maximo 55 videos por tanda.
- Maximo 2048 MB por archivo subido.

Puedes cambiar los limites antes de arrancar:

```powershell
$env:WHISPERMAX_MAX_BATCH_FILES = "3"
$env:WHISPERMAX_MAX_UPLOAD_MB = "1024"
$env:WHISPERMAX_FAST_THREADS = "8"
python main.py
```

## Cancelacion

Puedes cancelar cualquier video pendiente o en proceso desde la tabla de cola.

- Si esta en cola, se elimina antes de empezar.
- Si esta extrayendo audio, se corta `ffmpeg`.
- Si esta transcribiendo, se detiene en el siguiente avance interno de Whisper.
- En todos los casos se limpian el video temporal y el WAV temporal.

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

Para parar un servidor que se haya arrancado en segundo plano:

```powershell
Stop-Process -Id (Get-Content salidas\server.pid)
```
