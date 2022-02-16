# app.py
from flask import Flask, request, jsonify
import flask
import conv
import torch
import torch.nn as nn
from torch.autograd import Variable
import pyworld as pw
import soundfile as sf
import numpy as np
import argparse
import os.path
from lstm import MyBLSTM
from pydub import AudioSegment
import werkzeug

app = Flask(__name__)

### POST METHOD WHICH RECIEVES THE WHISPERED FILE AND CONVERTS IT

@app.route('/post', methods=['POST'])
def test():

    ### PROCESS THE REQUEST BODY

    file = request.files['file']
    gender = request.form['gender']

    ### EXTRACT THE FILENAME AND SAVE THE FILE ON SERVER

    filename = werkzeug.utils.secure_filename(file.filename)
    print("\n rec file is called: " + file.filename)    # DEB FOR SERVER LOGS
    print(gender)                                       # DEB FOR SERVER LOGS
    file.save(filename)

    ### CONVERT THE FILE INTO SOUNDFILE READABLE WAV CODEC AND CONTAINER

    sound = AudioSegment.from_file(filename)
    sound.export("coded.wav", format="wav")

    ### APPLY THE CONVERSION

    x, fs = conv.main("coded.wav", gender=gender)

    ### SAVE THE CONVERTED FILE

    sf.write("aux.wav", x, fs)

    ### RESPOND TO THE CLIENT, THAT THE CONVERSION WENT WELL

    return "CONVERSION OK"

@app.route('/get_conv', methods=['GET'])
def get_conv():

    ### SIMPLY RETURN THE CONVERTED FILE ON DEMAND.

    return flask.send_file("aux.wav", mimetype="application/octet-stream", attachment_filename="aux.wav")

if __name__ == '__main__':

    ### RUN WITH THREADED OPTION ON.

    app.run(threaded=True, port=5000)
