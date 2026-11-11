import glob

import gradio as gr
import tensorflow as tf

from preprocess import CLASS_NAMES, preprocess

model = tf.keras.models.load_model("model.keras")

# Contoh gambar dari test set Fashion-MNIST (satu per kelas, diperbesar 10x supaya terlihat)
EXAMPLES = sorted(glob.glob("examples/*.png"))


def predict(img):
    if img is None:
        return None, None
    x = preprocess(img)
    if x is None:
        return None, None
    probs = model.predict(x, verbose=0)[0]
    scores = {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)}
    preview = (x[0, :, :, 0] * 255).round().astype("uint8")
    return scores, preview


with gr.Blocks(title="Fashion-MNIST Classifier") as demo:
    gr.Markdown(
        "# Fashion-MNIST Classifier\n"
        "Upload satu gambar pakaian (atau klik salah satu contoh), lalu klik **Predict**.\n\n"
        "Hasil paling bagus kalau gambarnya berisi satu pakaian dengan latar polos (hitam atau putih)."
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", sources=["upload", "clipboard"], label="Gambar pakaian", height=280)
            with gr.Row():
                clear_btn = gr.Button("Clear")
                predict_btn = gr.Button("Predict", variant="primary")
            gr.Examples(examples=EXAMPLES, inputs=image, label="Contoh dari test set Fashion-MNIST")
        with gr.Column():
            label = gr.Label(num_top_classes=3, label="Prediksi")
            preview = gr.Image(label="Input yang dilihat model (28x28)", image_mode="L", width=140, height=140)

    predict_btn.click(predict, inputs=image, outputs=[label, preview])
    clear_btn.click(lambda: (None, None, None), outputs=[image, label, preview])

if __name__ == "__main__":
    demo.launch()
