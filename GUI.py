import tkinter as tk
import numpy as np
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

finalModel=load_model('finalModel.h5')

classes = {4: ('nv', 'melanocytic nevi'),
           6: ('mel', 'melanoma'),
           2 :('bkl', 'benign keratosis-like lesions'), 
           1:('bcc' , 'basal cell carcinoma'),
           5: ('vasc', 'pyogenic granulomas and hemorrhage'),
           0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
           3: ('df', 'dermatofibroma')}


def load_image(file_path):
    image = Image.open(file_path)
    image = image.resize((28, 28))       
    return image

def predict_image():
    imagearr = np.array(image)
    imagearr = np.expand_dims(imagearr, axis=0) 
    prediction=finalModel.predict(imagearr)
    output=np.argmax(prediction)
    print(classes[output][1])
    label_predict.config(text=classes[output][1])
    label_predict.text = classes[output][1]
    


def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        global image
        image = load_image(file_path)
        display_image(file_path)

def display_image(file_path):
    # image = Image.open(file_path)
    imagedisp=image.resize((244,244))
    imagedisp.thumbnail((300, 300))
    photo = ImageTk.PhotoImage(imagedisp)
    label_image.config(image=photo)
    label_image.image = photo


def display_prediction(file_path, prediction):
    # img = Image.open(file_path)
    image.thumbnail((250, 250))
    image = ImageTk.PhotoImage(image)
    canvas.image = image
    canvas.create_image(0, 0, anchor='nw', image=image)

root = tk.Tk()
root.title("Skin Cancer Prediction System")
# root.eval('tk::PlaceWindow . center')

canvas = tk.Canvas(root, width=500, height=0)
canvas.pack()
# root.resizable(0,0)


label_image = tk.Label(root, text="")
label_image.pack()

label_predict = tk.Label(root, text="")
label_predict.pack()

predict_button = tk.Button(root, text="Predict Image", command=predict_image)
predict_button.pack(pady=10, side="bottom")

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=10, side="bottom")  

  
root.mainloop()
