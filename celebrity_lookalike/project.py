import os
from deepface import DeepFace
from tqdm import tqdm
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox
import  ast
import tkinter as tk
from tkinter import filedialog

######################################################################################################################################3

target_img_name = None

def browse_files():
    global target_img_name
    desktopname = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select a File", filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
    if desktopname:
        target_img_name = os.path.basename(desktopname)
        print("Selected file:", target_img_name)
        root.destroy()
        
        

root = tk.Tk()
root.title("File Browser")
root.geometry('300x300+300+300')
root.configure(bg="#fff")
root.resizable(False,False)

Label(root,text='Browse your photo',bg='#fff',font=('Calibri(body)',15,'bold')).pack(expand=True)


browse_button = tk.Button(root, text="Browse Files", command=browse_files)
browse_button.pack(pady=10)


root.mainloop()

# Now you can access the selected file path using the selected_file variable
if target_img_name:
    print("Selected file:", target_img_name)
    
    

############################################################################################################################################

characters = []
for dirpath, dirnames, filenames in os.walk ("celebrity_photo/"):
    for filename in filenames:
        if ".jpg" in filename:
          characters.append( dirpath + filename)
          

similarities = {}
for character in tqdm(characters):
    obj = DeepFace.verify(img1_path = character , img2_path = target_img_name ,
                          model_name = "Facenet" , detector_backend = "mtcnn" ,
                           distance_metric = "euclidean")
    similarities[character] = obj ["distance"]
    

df = pd.DataFrame(similarities.items(), columns = ["character" , "distance"])

df = df.sort_values(by=["distance"])




def openpage():
    
    # Retrieve the closest match
    closest_match = df.iloc[0]

    # Load celebrity and target images
    celeb_img_path = closest_match["character"]
    target_img_path = target_img_name

    # Perform face detection
    celebrity_img = DeepFace.detectFace(img_path=celeb_img_path, detector_backend="mtcnn")
    target_img = DeepFace.detectFace(img_path=target_img_path, detector_backend="mtcnn")

    # Calculate similarity percentage
    similarity_percentage = 100 - closest_match["distance"]

    # Plot images
    fig = plt.figure(figsize=(10, 10))

    ax1 = fig.add_subplot(1, 2, 1)
    plt.imshow(celebrity_img)
    plt.axis("off")
    plt.title(os.path.basename(celeb_img_path))

    ax2 = fig.add_subplot(1, 2, 2)
    plt.imshow(target_img)
    plt.axis("off")
    plt.title(os.path.basename(target_img_path))

    # Add text box for percentage
    plt.text(0.5, 0.05, f"{similarity_percentage:.2f}% Match", ha="center", fontsize=12, transform=fig.transFigure)

    plt.show()
    window.quit()



window=Tk()
window.title('which celebrity do you look-alike')
window.geometry('300x300+300+300')
window.configure(bg="#fff")
window.resizable(False,False)

Label(window,text='Which celebrity you look-alike',bg='#fff',font=('Calibri(body)',15,'bold')).pack(expand=True)

button = Button(text="Match",height=2,command=openpage)
button.pack()
window.mainloop()


