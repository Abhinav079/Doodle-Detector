import pickle
import os.path

import tkinter.messagebox
from tkinter import *
from tkinter import simpledialog, filedialog

import PIL
import PIL.Image, PIL.ImageDraw
import cv2 as cv
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class DrawingClassifier:

    def __init__(self) -> None:
        self.class1, self.class2, self.class3 = None, None, None
        self.class1_counter, self.class2_counter, self.class3_counter = None, None, None
        self.clf = None
        self.proj_name = None
        self.root = None
        self.image1 = None
        
        self.status_label = None
        self.canvas = None
        self.draw = None

        self.brush_width = 15
        self.classes_prompt()
        self.init_gui()

    def classes_prompt(self):
        msg = Tk()
        msg.withdraw()

        self.proj_name = simpledialog.askstring("Project Name", "Please enter your project name down below!", parent=msg)
        if os.path.exists(self.proj_name):
            pickle_file_path = os.path.join(self.proj_name, f'{self.proj_name}_data.pickle')
            with open(pickle_file_path, 'rb') as f:
                data = pickle.load(f)
            self.class1 = data['c1']
            self.class2 = data['c2']
            self.class3 = data['c3']                           
            self.class1_counter = data['c1c']                                            
            self.class2_counter = data['c2c']                
            self.class3_counter = data['c3c']      
            self.clf = data['clf']

        else:
            self.class1 = simpledialog.askstring('Class1', 'What is the first class called?', parent= msg)        
            self.class2 = simpledialog.askstring('Class2', 'What is the first class called?', parent= msg)        
            self.class3 = simpledialog.askstring('Class3', 'What is the first class called?', parent= msg)
            
            self.class1_counter = 1
            self.class2_counter = 1
            self.class3_counter = 1

            self.clf = LinearSVC()

            os.mkdir(self.proj_name)
            os.chdir(self.proj_name)
            os.mkdir(self.class1)
            os.mkdir(self.class2)
            os.mkdir(self.class3)
            os.chdir("..")

    def init_gui(self):                  
        WIDTH = 500
        HEIGHT = 500
        WHITE = (255,255,255)           

        self.root = Tk()
        self.root.title(f'Doodle Detect - {self.proj_name}')

        self.canvas = Canvas(self.root, width=WIDTH, height=HEIGHT, bg='white')
        self.canvas.pack(expand=YES, fill=BOTH)
        self.canvas.bind('<B1-Motion>',self.paint)

        self.image1 = PIL.Image.new('RGB',(WIDTH, HEIGHT), WHITE)
        self.draw = PIL.ImageDraw.Draw(self.image1)

        btn_frame = tkinter.Frame(self.root)
        btn_frame.pack(fill=X, side=BOTTOM)

        btn_frame.columnconfigure(0, weight=1)        
        btn_frame.columnconfigure(1, weight=1)                
        btn_frame.columnconfigure(2, weight=1)                

        class1_btn = Button(btn_frame, text=self.class1, command= lambda: self.save(1))        
        class1_btn.grid(row=0, column=0, sticky= W+E)

        class2_btn = Button(btn_frame, text=self.class2, command= lambda: self.save(2))        
        class2_btn.grid(row=0, column=1, sticky= W+E)

        class1_btn = Button(btn_frame, text=self.class3, command= lambda: self.save(3))        
        class1_btn.grid(row=0, column=2, sticky= W+E)

        bm_btn = Button(btn_frame, text='Brush-', command=self.brushminus )
        bm_btn.grid(row=1, column=0, sticky=W+E)

        clear_btn = Button(btn_frame, text='clear', command=self.clear )
        clear_btn.grid(row=1, column=1, sticky=W+E)

        bp_btn = Button(btn_frame, text='Brush+', command=self.brushplus )
        bp_btn.grid(row=1, column=2, sticky=W+E)

        train_btn = Button(btn_frame, text='Train Model', command=self.train_model )
        train_btn.grid(row=2, column=0, sticky=W+E)

        save_btn = Button(btn_frame, text='Save Model', command=self.save_model )
        save_btn.grid(row=2, column=1, sticky=W+E)        

        load_btn = Button(btn_frame, text='Load Model', command=self.load_model )
        load_btn.grid(row=2, column=2, sticky=W+E)

        change_btn = Button(btn_frame, text='Change Model', command=self.rotate_model )
        change_btn.grid(row=3, column=0, sticky=W+E)

        predict_btn = Button(btn_frame, text='Predict', command=self.predict )
        predict_btn.grid(row=3, column=1, sticky=W+E)        

        save_everything_btn = Button(btn_frame, text='Save Everything', command=self.save_everything )
        save_everything_btn.grid(row=3, column=2, sticky=W+E)        

        self.status_label = Label(btn_frame, text=f'Current Model: {type(self.clf).__name__}')
        self.status_label.config(font=('Arial',10))
        self.status_label.grid(row=4, column=1, sticky=W+E)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.attributes("-topmost",True)
        self.root.mainloop()

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill='black', width=self.brush_width)
        self.draw.rectangle([x1, y1, x2 + self.brush_width, y2 + self.brush_width], fill='black', width=self.brush_width)

    def save(self, class_num):
        self.image1.save('temp.png')
        img = PIL.Image.open('temp.png')
        img.thumbnail((50, 50), PIL.Image.ANTIALIAS)
    
        if class_num == 1:
            img.save(f'{self.proj_name}/{self.class1}/{self.class1_counter}.png', 'PNG')
            self.class1_counter += 1
    
        elif class_num == 2:
            img.save(f'{self.proj_name}/{self.class2}/{self.class2_counter}.png', 'PNG')
            self.class2_counter += 1
    
        elif class_num == 3:
            img.save(f'{self.proj_name}/{self.class3}/{self.class3_counter}.png', 'PNG')
            self.class3_counter += 1
    
        # Clear the canvas after saving
        self.clear()

         

    def brushminus(self):
        if self.brush_width > 1:
            self.brush_width -= 1
    def brushplus(self):
        self.brush_width += 1

    def clear(self):
        self.canvas.delete('all')
        self.draw.rectangle([0,0,1000,1000], fill='white')

    def train_model(self):
        img_list = np.array([])
        class_list = np.array([])
    
        for x in range(1, self.class1_counter):
            img = cv.imread(f'{self.proj_name}/{self.class1}/{x}.png')[:,:,0]
            img = img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 1)

        for x in range(1, self.class2_counter):
            img = cv.imread(f'{self.proj_name}/{self.class2}/{x}.png')[:,:,0]
            img = img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 2)

        for x in range(1, self.class3_counter):
            img = cv.imread(f'{self.proj_name}/{self.class3}/{x}.png')[:,:,0]
            img = img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 3)

        img_list = img_list.reshape(self.class1_counter-1 + self.class2_counter - 1 + self.class3_counter - 1, 2500)
        self.clf.fit(img_list, class_list)
        tkinter.messagebox.showinfo('Doodle Detect', 'Model Successfully trained', parent=self.root)


    def predict(self):
        self.image1.save('temp.png')
        img = PIL.Image.open('temp.png')
        img.thumbnail((50,50), PIL.Image.ANTIALIAS)
        img.save('predctshape.png','PNG')

        img = cv.imread('predctshape.png')[:,:,0]
        img = img.reshape(2500)
        prediction = self.clf.predict([img])
        if prediction[0] == 1:
            tkinter.messagebox.showinfo('Doodle Detect Drawing Classifier', f'The drawing is prbably a {self.class1}',parent = self.root)
        elif prediction[0] == 2:
            tkinter.messagebox.showinfo('Doodle Detect Drawing Classifier', f'The drawing is prbably a {self.class2}',parent = self.root)
        elif prediction[0] == 3:
            tkinter.messagebox.showinfo('Doodle Detect Drawing Classifier', f'The drawing is prbably a {self.class3}',parent = self.root)
  

    def rotate_model(self):
        if isinstance(self.clf, LinearSVC):
            self.clf = KNeighborsClassifier()
        elif isinstance(self.clf, KNeighborsClassifier):
            self.clf = LogisticRegression()
        elif isinstance(self.clf, LogisticRegression):
            self.clf = DecisionTreeClassifier()            
        elif isinstance(self.clf, DecisionTreeClassifier):
            self.clf = RandomForestClassifier()
        elif isinstance(self.clf, RandomForestClassifier):
            self.clf = GaussianNB()            
        elif isinstance(self.clf, GaussianNB):
            self.clf = LinearSVC()
        self.status_label.config(text = f'Current Model: {type(self.clf).__name__}')    

    def save_model(self):
        # Ask the user for the file path with a default extension
        file_path = filedialog.asksaveasfilename(defaultextension='.pickle')
    
        # Check if the user canceled the file dialog
        if not file_path:
            return
    
        # Save the model to the specified file
        with open(file_path, 'wb') as f:
            pickle.dump(self.clf, f)
    
        # Display a success message
        tkinter.messagebox.showinfo('Doodle Defect Drawing Classifier', 'Model Successfully Saved', parent=self.root)


    def load_model(self):
        file_path = filedialog.askopenfilename()
        with open(file_path, 'rb') as f:
            self.clf = pickle.load(f)
        tkinter.messagebox.showinfo('Doodle Defect Drawing Classifier','Model Successfully loaded', parent=self.root)    


    def save_everything(self):
        data = {'c1': self.class1, 'c2': self.class2, 'c3': self.class3, 'c1c': self.class1_counter, 'c2c': self.class1_counter,
                'c3c': self.class1_counter, 'clf': self.clf, 'pname': self.proj_name}
        with open(f'{self.proj_name}/{self.proj_name}_data.pickle','wb') as f:
            pickle.dump(data, f)
        tkinter.messagebox.showinfo('Doodle Defect Drawing Classifier','Project Successfully Saved', parent=self.root)    
                  

    def on_closing(self):
        answer =  tkinter.messagebox.askokcancel("Quit", "Do you want to Save your work?",parent = self.root)
        if answer is not None:
            if answer:
               self.save_everything()
            self.root.destroy()
            exit()   


DrawingClassifier()
