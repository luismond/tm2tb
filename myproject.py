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
from flask import after_this_request
from werkzeug.utils import secure_filename
from tm2tb import tm2tb_main
from flask import send_from_directory

# App config.
DEBUG = False
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.csv', '.mxliff','.mqxliff', '.tmx']
app.config['UPLOAD_PATH'] = ""
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
    
@app.route('/<filename>')
def uploaded_file(filename):
    filepath = os.path.join(app.config['UPLOAD_PATH'], filename)
    @after_this_request
    def remove_file(response): 
        os.remove(filepath)
        return response
    return send_from_directory(filepath)

@app.route("/prev")
def prev(filename):
    result_html = Markup(tm2tb_main(filename))
    flash(result_html)
    @after_this_request
    def remove_file(response):
        filepath = os.path.join(app.config['UPLOAD_PATH'], filename)
        os.remove(filepath)
        return response
    return render_template('index.html')
 
@app.route("/<filename>")
def get_file(filename):
    redirect(url_for('uploaded_file', filename=filename))
    return 
            
if __name__ == "__main__":
	app.run(host='0.0.0.0')
