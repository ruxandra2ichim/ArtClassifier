import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner('art_model.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Art Classifier"
description = "A toy art classifier trained on a small database for the FastAI Course. Created as a demo for Gradio and HuggingFace Spaces."
interpretation='default'
enable_queue=True

gr.Interface(fn=predict,inputs=gr.inputs.Image(shape=(512, 512)),outputs=gr.outputs.Label(num_top_classes=3),\
    title=title,description=description,interpretation=interpretation,enable_queue=enable_queue).launch()