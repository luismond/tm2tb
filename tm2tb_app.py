#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tm2tb client app
"""
import os
import sys
from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from flask import url_for
from flask import abort
from flask import flash
from flask import Markup
from werkzeug.utils import secure_filename
from tm2tb import tm2tb_main
from flask import send_from_directory

# App config.
DEBUG = False
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.csv', '.mxliff','.mqxliff', '.tmx']
app.config['UPLOAD_PATH'] = "tmp"
app.config['SECRET_KEY'] = 'secret'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def post_file():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        else:
            uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    return prev(filename)
    
@app.route('/tmp/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

@app.route("/prev")
def prev(filename):
     result_html = Markup(tm2tb_main(filename))
     flash(result_html)
     return render_template('index.html')
 
@app.route("/<filename>")
def get_file(filename):
    return redirect(url_for('uploaded_file', filename=filename))
            
if __name__ == "__main__":
	app.run(host='0.0.0.0', port='5002')
