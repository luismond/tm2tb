#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TM2TB web application.
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
from flask import send_from_directory
from werkzeug.utils import secure_filename
from tm2tb import tm2tb_main

# App config.
DEBUG = False
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.csv', '.mxliff','.mqxliff', '.tmx']
app.config['UPLOAD_PATH'] = "uploads"
app.config['SECRET_KEY'] = 'secret'

@app.route('/')
#Show the homepage
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def post_file():
    #Get the uploaded file objet from the upload form
    uploaded_file = request.files['file']
    #Get the file name
    filename = secure_filename(uploaded_file.filename)
    #If the file name is empty, return "please select a file"
    if filename == '':
        flash(Markup('<p class="text-center error">Please select a file</p>'))
        return redirect(url_for('index'))
    #If the file name exists, proceed
    if filename != '':
        file_extension = os.path.splitext(filename)[1]
        #If file extension is not supported, return "File not supported"
        if file_extension not in app.config['UPLOAD_EXTENSIONS']:
            flash(Markup('<p class="text-center error">File not supported!</p>'))
            return redirect(url_for('index'))
        #If file extensions is supported, temporarily save file and run main tm2tb process
        else:
            uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
            return preview(filename)

@app.route("/")
def preview(filename):
    '''
    After getting a filename from a validated file, run main tm2tb process, 
    which will return an html table showing the user's text and the terms extracted (or a file error message)
    '''
    result_html = Markup(tm2tb_main(filename))
    #Show the html result
    flash(result_html)
    @after_this_request
    #After this request, delete the user's file
    def remove_file(response):
        os.remove(os.path.join(app.config['UPLOAD_PATH'], filename))
        return response
    return render_template('index_results.html')


@app.route('/uploads/<filename_tb>')
def uploaded_file(filename_tb):
    #When the user clicks "download", send him the final glossary and delete it afterwards
    @after_this_request
    def remove_file_tb(response):
        os.remove(os.path.join(app.config['UPLOAD_PATH'], filename_tb))
        return response
    return send_from_directory(app.config['UPLOAD_PATH'], filename_tb)

if __name__ == "__main__":
	app.run(host='0.0.0.0')
