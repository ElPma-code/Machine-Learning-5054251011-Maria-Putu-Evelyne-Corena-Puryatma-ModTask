import modal

app = modal.App("fashion-classifier")

# tensorflow==2.20.0 disamakan dengan versi yang dipakai saat melatih model di notebook
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("tensorflow==2.20.0", "gradio==6.31.0", "fastapi[standard]==0.162.0", "pillow")
    .add_local_file("model.keras", "/root/model.keras")
    .add_local_file("preprocess.py", "/root/preprocess.py")
    .add_local_file("gradio_app.py", "/root/gradio_app.py")
    .add_local_dir("examples", "/root/examples")
)


@app.function(
    image=image,
    max_containers=1,       # Gradio membutuhkan sticky session
    scaledown_window=300,   # container dimatikan setelah 5 menit tanpa request
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    import os
    import sys

    import gradio as gr
    from fastapi import FastAPI

    sys.path.insert(0, "/root")
    os.chdir("/root")
    from gradio_app import demo

    return gr.mount_gradio_app(app=FastAPI(), blocks=demo.queue(), path="/")
