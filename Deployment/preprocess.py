import numpy as np
from PIL import Image, ImageOps

# Urutan nama kelas sama dengan label 0-9 di Fashion-MNIST
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def preprocess(img: Image.Image):
    img = img.convert("L")
    arr = np.array(img)

    # Gambar Fashion-MNIST berupa pakaian terang di latar hitam.
    # Kalau latar gambarnya terang (misalnya foto produk berlatar putih), warnanya dibalik dulu.
    tepi = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if np.median(tepi) > 127:
        img = ImageOps.invert(img)
        arr = np.array(img)

    # Cari area yang berisi pakaian
    ys, xs = np.nonzero(arr > 20)
    if len(xs) == 0:
        return None  # gambar kosong

    # Crop sekitar pakaian, lalu skala sisi terpanjang ke 28 px.
    # Filter BOX (rata-rata piksel) dipakai supaya gambar tidak jadi blur saat diperkecil.
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    item = img.crop(box)
    scale = 28 / max(item.size)
    new_size = (max(1, round(item.width * scale)), max(1, round(item.height * scale)))
    item = item.resize(new_size, Image.Resampling.BOX)

    # Letakkan di tengah kanvas hitam 28x28. Kalau sisa ruangnya ganjil, 1 piksel lebihnya
    # ditaruh di kiri/atas, sama seperti posisi gambar di dataset Fashion-MNIST.
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(item, ((28 - new_size[0] + 1) // 2, (28 - new_size[1] + 1) // 2))

    # Model dilatih dengan piksel 0-1, jadi nilainya dibagi 255
    return np.array(canvas, dtype="float32").reshape(1, 28, 28, 1) / 255.0
