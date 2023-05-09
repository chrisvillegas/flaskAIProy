from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok

import torch
from diffusers import StableDifussionPipeline

import base64
from io import BytesIO

#load model
pipe = StableDifussionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", revision = "fp16", torch_dtype=torch.float16
)

pipe.to("cuda")

#Start flask app and set ngrok
app = Flask(__name__)
run_with_ngrok(app)

@app.route("/")
def.initial():
    return render_template("index.html")

@app.route("/submit-prompt", methouds=["POST"])
def generate_image():
        prompt = request.from["prompt-input"]

        image = pipe(prompt).images[0]

        buffered =  BytesIO()
        image.save(buffered, format = "PNG")
        img_str = base64.b64encode(buffered.getvalue())
        img_str = "data:image/png;base64," + str(img_str)[2:-1]

        return render_template("index.html", generated_image=img_str)

if __name__ == "__main__":
    app.run()
