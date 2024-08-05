from tkinter import *
from tkinter import messagebox
import  ast

root=Tk()
root.title('Login')
root.geometry('925x500+300+200')
root.configure(bg="#fff")
root.resizable(False,False)

def signin():
    username=user.get()
    password=code.get()
    
    file=open('datasheet.txt','r')
    d=file.read()
    r=ast.literal_eval(d)
    file.close()
    
   # print(r.keys())
   # print(r.values())
    
    if username in r.keys() and password==r[username]:
       import os
       import pandas as pd
       import cv2
       import matplotlib.pyplot as plt
       from tkinter import messagebox, filedialog
       import tkinter as tk
       from deepface import DeepFace
       from tqdm import tqdm

       def browse_files():
           global target_img_name
           desktopname = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select a File", filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
           if desktopname:
               target_img_name = desktopname
               print("Selected file:", target_img_name)
               root.destroy()
           else:
               messagebox.showinfo("Info", "No file selected!")

       def find_celebrity_lookalike(target_img_name):
           if not target_img_name:
               messagebox.showinfo("Error", "Please select an image first!")
               return

           characters = []
           for dirpath, dirnames, filenames in os.walk("celebrity_photo/"):
               for filename in filenames:
                   if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                       characters.append(os.path.join(dirpath, filename))

           if not characters:
               messagebox.showinfo("Error", "No celebrity images found!")
               return

           similarities = {}
           for character in tqdm(characters, desc="Processing"):
               obj = DeepFace.verify(img1_path=character, img2_path=target_img_name,
                                     model_name="Facenet", detector_backend="mtcnn",
                                     distance_metric="euclidean")
               similarities[character] = obj["distance"]

           df = pd.DataFrame(similarities.items(), columns=["character", "distance"])
           df = df.sort_values(by=["distance"])

           return df

       def openpage():
           df = find_celebrity_lookalike(target_img_name)

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

# GUI for selecting an image file
       root = tk.Tk()
       root.title("File Browser")
       root.geometry('300x300+300+300')
       root.configure(bg="#fff")
       root.resizable(False, False)

       Label(root, text='Browse your photo', bg='#fff', font=('Calibri(body)', 15, 'bold')).pack(expand=True)

       browse_button = tk.Button(root, text="Browse Files", command=browse_files)
       browse_button.pack(pady=10)

       root.mainloop()

# GUI for displaying the match
       if target_img_name:
           window = Tk()
           window.title('Which celebrity do you look-alike')
           window.geometry('300x200+300+300')
           window.configure(bg="#fff")
           window.resizable(False, False)

           Label(window, text='Click the button to find your celebrity look-alike!', bg='#fff', font=('Calibri(body)', 15, 'bold')).pack(expand=True)

           button = Button(window, text="Match", height=2, command=openpage)
           button.pack()

           window.mainloop()

        
    else:
        messagebox.showerror('Invalid','Invalid username or password')
  

    
def signup_command():
    window=Toplevel(root)
    window.title('Login')
    window.geometry('925x500+300+200')
    window.configure(bg="#fff")
    window.resizable(False,False)

    def signup():
        username=user.get()
        password=code.get()
        confirm_password=conform_code.get()
    
        if password==confirm_password:
            try:
                file=open('datasheet.txt','r+')
                d=file.read()
                r=ast.literal_eval(d)
            
                dict2={username:password}
                r.update(dict2)
                file.truncate(0)
                file.close()
            
                file=open('datasheet.txt','w')
                w=file.write(str(r))
            
                messagebox.showinfo('Signup','Sucessfully sign up')
                window.destroy()
                            
            except:
                file=open('datasheet.txt','w')
                pp=str({'username':'password'})
                file.write(pp)
                file.close()
            
        else:
            messagebox.showerror('Invalid','Both password should match')
        
    def sign():
        window.destroy()  
    
    img = PhotoImage(file='register.png')
    Label(window,image=img,bg='white').place(x=50,y=90)

    frame =Frame(window,width=350,height=390,bg='Light Blue')
    frame.place(x=480,y=50)

    heading=Label(frame,text='Sign Up',fg='dark blue',bg='Light Blue',font=('Microsoft YaHei UI Light',23,'bold'))
    heading.place(x=120,y=5)

    def on_enter(e):
        user.delete(0,'end')
    
    def on_leave(e):
        if user.get()=='':
            user.insert(0,'Username')
        
    user = Entry(frame,width=25,fg='black',border=0,bg='light blue',font=('Microsoft YaHei UI Light',11))
    user.place(x=30,y=80)
    user.insert(0,'Username')
    user.bind('<FocusIn>',on_enter)
    user.bind('<FocusOut>',on_leave)

    Frame(frame,width=295,height=2,bg='black').place(x=25,y=107)

    def on_enter(e):
        code.delete(0,'end')
    
    def on_leave(e):
        if code.get()=='':
            code.insert(0,'Password')
        
    code = Entry(frame,width=25,fg='black',border=0,bg='light blue',font=('Microsoft YaHei UI Light',11))
    code.place(x=30,y=150)
    code.insert(0,'Password')
    code.bind('<FocusIn>',on_enter)
    code.bind('<FocusOut>',on_leave)

    Frame(frame,width=295,height=2,bg='black').place(x=25,y=177)

    def on_enter(e):
        conform_code.delete(0,'end')
    
    def on_leave(e):
        if conform_code.get()=='':
            conform_code.insert(0,'Confirm Password')
        
    conform_code = Entry(frame,width=25,fg='black',border=0,bg='light blue',font=('Microsoft YaHei UI Light',11))
    conform_code.place(x=30,y=220)
    conform_code.insert(0,'Confirm Password')
    conform_code.bind('<FocusIn>',on_enter)
    conform_code.bind('<FocusOut>',on_leave)

    Frame(frame,width=295,height=2,bg='black').place(x=25,y=247)

    Button(frame,width=39,pady=7,text='Sign up',bg='#57a1f8',fg='white',border=0,command=signup).place(x=35,y=280)
    label=Label(frame,text="I have an account?",fg='dark blue',bg='light blue',font=('Microsoft YaHei UI Light',9))
    label.place(x=90,y=340)

    signin = Button(frame,width=6,text='Sign In',border=0,bg='#57a1f8',cursor='hand2',fg='dark blue',command=sign)
    signin.place(x=200,y=340)

    window.mainloop()


img = PhotoImage(file='login.png')
Label(root,image=img,bg='white').place(x=50,y=50)

frame =Frame(root,width=350,height=350,bg='Light Blue')
frame.place(x=480,y=70)

heading=Label(frame,text='Sign In',fg='dark blue',bg='Light Blue',font=('Microsoft YaHei UI Light',23,'bold'))
heading.place(x=120,y=5)

def on_enter(e):
    user.delete(0,'end')
    
def on_leave(e):
    name=user.get()
    if name=='':
        user.insert(0,'Username')
        
user = Entry(frame,width=25,fg='black',border=0,bg='light blue',font=('Microsoft YaHei UI Light',11))
user.place(x=30,y=80)
user.insert(0,'Username')
user.bind('<FocusIn>',on_enter)
user.bind('<FocusOut>',on_leave)

Frame(frame,width=295,height=2,bg='black').place(x=25,y=107)

def on_enter(e):
    code.delete(0,'end')
    
def on_leave(e):
    name=code.get()
    if name=='':
        code.insert(0,'Password')
        
code = Entry(frame,width=25,fg='black',border=0,bg='light blue',font=('Microsoft YaHei UI Light',11))
code.place(x=30,y=150)
code.insert(0,'Password')
code.bind('<FocusIn>',on_enter)
code.bind('<FocusOut>',on_leave)

Frame(frame,width=295,height=2,bg='black').place(x=25,y=177)

Button(frame,width=39,pady=7,text='Sign in',bg='#57a1f8',fg='white',border=0,command=signin).place(x=35,y=204)
label=Label(frame,text="Don't have an account?",fg='dark blue',bg='light blue',font=('Microsoft YaHei UI Light',9))
label.place(x=50,y=265)

sign_up = Button(frame,width=6,text='Sign Up',border=0,bg='#57a1f8',cursor='hand2',fg='dark blue',command=signup_command)
sign_up.place(x=200,y=265)



root.mainloop()
