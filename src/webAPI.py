from flask import Flask, Response, jsonify
import cv2
import mediapipe as mp
import pickle
import numpy as np
import createDataset


app = Flask(__name__)


@app.route("/")
def test():
    return "Yes"


if __name__ == "__main__":
    app.run(debug=True)
