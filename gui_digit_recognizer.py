from keras.models import load_model
from tkinter import *
import tkinter as tk
from PIL import Image, ImageGrab
import numpy as np

model = load_model('mnist.h5')

def preprocess_image(img):
    # Resize image to 28x28 pixels
    img = img.resize((28, 28))
    # Convert to grayscale
    img = img.convert('L')
    img = np.array(img)
    # Normalize and reshape for model input
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img /= 255.0
    return img

def predict_digit(img):
    img = preprocess_image(img)
    res = model.predict(img)
    return np.argmax(res), max(res[0])

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Creating elements
        self.canvas = tk.Canvas(self, width=500, height=500, background='white', cursor='cross')
        self.label = tk.Label(self, text="Draw..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognize", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        x, y = (self.canvas.winfo_rootx(), self.canvas.winfo_rooty())
        width, height = (self.canvas.winfo_width(), self.canvas.winfo_height())
        a, b, c, d = (x, y, x + width, y + height)
        im = ImageGrab.grab(bbox=(a, b, c, d))

        digit, acc = predict_digit(im)
        self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')

app = App()
app.mainloop()
