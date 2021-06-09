from flask import Flask
from flask import Flask, flash, redirect, render_template, request, session
from flask.helpers import url_for
from werkzeug.utils import secure_filename
import os
import urllib.request
import torch
from fastai.vision import *
from fastai.metrics import error_rate, accuracy

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index(name=None):
    return render_template('index.html', name=name)


@app.route('/uploadimg')
def uploadimg():
    return render_template('upload.html')


@app.route('/uploadingimg', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('upload.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    learn = load_learner("dataset_with_mask")
    img = open_image('static/uploads/'+filename)
    # (image.jpg is any random image.)
    #img.show(figsize=(3, 3))
    pred_class, preds_idx, outputs = learn.predict(img)

    print("Class is : ", pred_class)
    print("Outputs is : ", outputs[preds_idx])
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('adminHome'))
    return render_template('login.html', error=error)


@app.route('/admin_home')
def adminHome():
    return render_template('admin_home.html')


@app.route('/uploadedimg', methods=['GET', 'POST'])
def uploadedimg():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            image.save(os.path.join(
                app.config["IMAGE_UPLOADS"], image.filename))
            return render_template("upload_image.html", uploaded_image=image.filename)
    return render_template("upload_image.html")


@app.route('/training')
def training():
    import training
    return render_template('training.html')


@app.route('/livedetect')
def livedetect(name=None):
    import face_recognize
    print("done")
    return render_template('index.html', name=name)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    app.debug = True
