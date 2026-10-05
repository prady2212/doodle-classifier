import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image

CLASSES = ['banana','hockey stick','squirrel','watermelon','bathtub',
           'laptop','rain','pond','nose','panda']
model = tf.keras.models.load_model('models/best_model.keras')

def preprocess(img):
    img = Image.fromarray(np.array(img).astype('uint8'))
    if img.mode == 'RGBA':                       # flatten onto white
        bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    g = 255 - np.array(img.convert('L'))         # strokes -> white on black
    ys, xs = np.where(g > 30)
    if len(xs) == 0:
        return None
    g = g[ys.min():ys.max()+1, xs.min():xs.max()+1]   # crop to drawing
    h, w = g.shape
    s = max(h, w)
    sq = np.zeros((s, s), dtype=np.uint8)
    sq[(s-h)//2:(s-h)//2+h, (s-w)//2:(s-w)//2+w] = g
    small = np.array(Image.fromarray(sq).resize((20, 20), Image.LANCZOS))
    out = np.zeros((28, 28), dtype=np.uint8)
    out[4:24, 4:24] = small                      # centered with padding
    return out.astype('float32') / 255.0

def predict(sketch):
    if sketch is None:
        return {}
    img = sketch['composite'] if isinstance(sketch, dict) else sketch
    x = preprocess(img)
    if x is None:
        return {}
    probs = model.predict(x.reshape(1, 28, 28, 1), verbose=0)[0]
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(label="Draw here", brush=gr.Brush(colors=["#000000"], default_size=8)),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Doodle Classifier",
    description="Draw one of: " + ", ".join(CLASSES),
    live=True,
)

if __name__ == "__main__":
    demo.launch()