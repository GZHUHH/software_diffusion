# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 19:28:45 2024

@author: huanghe
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import tkinter as tk 
import ttkbootstrap as ttk
from tkinter import filedialog# 文件对话框，用于打开和保存文件
from PIL import Image,ImageTk# 图像处理
import webbrowser# 打开网页
import joblib# 模型加载和保存
import sys# 系统操作
import os# 操作系统接口
from lightgbm import LGBMRegressor as lgb
from PIL import Image
from tkinter import font
import lightgbm as lgb
import numpy as np  # 导入 numpy 并设置别名为 np


model = joblib.load("../model/lgbm.pt") 

## The layout design of the main interface
root= tk.Tk()
root.title("Predict diffusivity  of material on stacking")
root.resizable(True,True) ## The window size can be changed
canvas1 = tk.Canvas(root, width = 820, height =730) ## Main window size
canvas1.pack()

## The plate layout within the main interface
re1=canvas1.create_rectangle(50,20,770,450)
re2=canvas1.create_rectangle(50,470,770,670)
re3=canvas1.create_rectangle(70,100,380,320,outline='orange')
re4=canvas1.create_rectangle(400,100,750,320,outline='orange')
re4=canvas1.create_rectangle(70,340,750,430,outline='orange')
re4=canvas1.create_rectangle(70,560,750,650,outline='orange')
label_B = tk.Label(root,font=('microsoft yahei',10),text='Predicted results')
canvas1.create_window(620, 340, window=label_B)
label_B = tk.Label(root,font=('microsoft yahei',10),text='Predicted results')
canvas1.create_window(620, 560, window=label_B)
label_B = tk.Label(root,font=('microsoft yahei',9),text='Author：He Huang ,Zhiwei Qiao,Guangzhou University')
canvas1.create_window(580, 715, window=label_B)

## Message box (Related literature on molecular physical properties)
def cmx1():
    window = tk.Tk()     
    window.title('Warm prompt')     
    window.geometry('350x250')
    link = tk.Label(window, text='The physical properties of gases \nare known from the literature:\nhttps://doi.org/10.1039/B802426J'
                    , font=('microsoft yahei',10),anchor="center")
    link.place(x=30, y=50) 
    def open_url(event):
        webbrowser.open("https://doi.org/10.1039/B802426J", new=0)         
    link.bind("<Button-1>", open_url)    
btn1=tk.Button(root, text='Tooltip',font=('microsoft yahei',10), command=cmx1, bg='deeppink', fg='light cyan')
btn1.configure(bg='coral')
canvas1.create_window(450, 140, window=btn1)

##Message box (Instructions for Prediction of a single material diffusiivity)
def resize(w, h, w_box, h_box, pil_image): 
  f1 = 1*w_box/w 
  f2 =1*h_box/h  
  factor = min([f1, f2])  
  width = int(w*factor)  
  height = int(h*factor)  
  return pil_image.resize((width, height), Image.Resampling.LANCZOS)
       
w_box = 600  
h_box = 450    

global tk_image 
photo1 = Image.open("../Img/full_name.png") 
w, h = photo1.size       
photo1_resized = resize(w, h, w_box, h_box, photo1)    
tk_image1 = ImageTk.PhotoImage(photo1_resized)

def cmx2():
    top2=tk.Toplevel() 
    top2.title('Instructions for use') 
    top2.geometry('620x500') 
    lab_1 = ttk.Label(top2,image=tk_image1) 
    lab_1.place(x=25, y=10) 
    top2.mainloop()  
  
btn2=tk.Button(root, text='README',font=('microsoft yahei',10), command=cmx2, bg='deeppink', fg='deeppink')
btn2.configure(bg='coral')
canvas1.create_window(120, 60, window=btn2)

## Message box (Instructions for batch Prediction of material diffusiivity)
def resize(w, h, w_box, h_box, pil_image):
  f1 = 1*w_box/w 
  f2 =1*h_box/h  
  factor = min([f1, f2])  
  width = int(w*factor)  
  height = int(h*factor)  
  return pil_image.resize((width, height), Image.Resampling.LANCZOS)     
w_box = 600  
h_box = 500    

global tk_image 
photo2 = Image.open("../Img/sample_file.png")  
w, h = photo2.size     
photo2_resized = resize(w, h, w_box, h_box, photo2)    
tk_image2 = ImageTk.PhotoImage(photo2_resized)

def cmx3():
    top1=tk.Toplevel()
    top1.title('Instructions for use')     
    top1.geometry('680x580')
    lab2 = tk.Label(top1, text='You need to create the data you want to compute\nin the format below (For example：the prediction \nof DC2H6):'
                    , font=('microsoft yahei',15),anchor="nw",justify='left')
    lab2.place(x=20, y=20) 
    lab3 = tk.Label(top1, text='After creating the file, you can click the import file \nbutton on the screen.The predicted result  will be \nsaved in "Result/Batch_Predicted_D.xlsx".'
                    , font=('microsoft yahei',15),anchor="nw",justify='left')
    lab3.place(x=30, y=450)
    lab3 = ttk.Label(top1,text="photo:",image=tk_image2)
    lab3.place(x=30, y=120) 
    top1.mainloop()     
btn3=tk.Button(root, text='README',font=('microsoft yahei',10),command=cmx3, bg='deeppink', fg='deeppink')
btn3.configure(bg='coral')
canvas1.create_window(120, 520, window=btn3)

## Sets the label and entry for entering the nine descriptor 
label_Z = tk.Label(root,font=('microsoft yahei',13),text='Predict diffusivity of material')
canvas1.create_window(415, 20, window=label_Z)

label_L = tk.Label(root,font=('microsoft yahei',11),text='Physical property of material')
canvas1.create_window(225, 100, window=label_L)

label1 = tk.Label(root,font=('microsoft yahei',10),text='φ：') ## create 1st label box 
canvas1.create_window(160, 140, window=label1)
entry1 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 1st entry box 
canvas1.create_window(300, 140, window=entry1)

label2 = tk.Label(root,font=('microsoft yahei',10), text='PLD (Å): ') ## create 2st label box 
canvas1.create_window(160, 180, window=label2)
entry2 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 2nd entry box
canvas1.create_window(300, 180, window=entry2)

label3 = tk.Label(root,font=('microsoft yahei',10), text='LCD (Å): ') ## create 3st label box 
canvas1.create_window(160, 220, window=label3)
entry3 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 3nd entry box
canvas1.create_window(300, 220, window=entry3)

label4 = tk.Label(root,font=('microsoft yahei',10,"italic"), text='ρ') ## create 4st label box 
canvas1.create_window(145, 260, window=label4)
l0 = tk.Label(root,font=('microsoft yahei',10), text='(kg/cm^3): ') 
canvas1.create_window(190, 260, window=l0)
entry4 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 4nd entry box
canvas1.create_window(300, 260, window=entry4)

label5 = tk.Label(root, font=('microsoft yahei',10),text='VSA (m^2/cm^3): ') ## create 5st label box 
canvas1.create_window(160, 300, window=label5) 
entry5 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 5nd entry box
canvas1.create_window(300, 300, window=entry5)

label_R = tk.Label(root,font=('microsoft yahei',11),text='Physical property of gas molecules') 
canvas1.create_window(575, 100, window=label_R)

label6 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Dia') ## create 1st 6abel box 
canvas1.create_window(490,180, window=label6)
l1 = tk.Label(root, font=('microsoft yahei',10),text='(Å):')
canvas1.create_window(525,180, window=l1) 
entry6 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 6nd entry box
canvas1.create_window(660,180, window=entry6)

label7 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Pol') ## create 7st label box 
canvas1.create_window(430,220, window=label7)
l2 = tk.Label(root, font=('microsoft yahei',10,),text='(×10^25 cm^3): ')
canvas1.create_window(520,220, window=l2)
entry7 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 7nd entry box
canvas1.create_window(660,220, window=entry7)

label8 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Dip') ## create 8st label box 
canvas1.create_window(430,260, window=label8) 
l3 = tk.Label(root, font=('microsoft yahei',10),text='(×10^18 cm): ')
canvas1.create_window(510,260, window=l3)  
entry8 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 8nd entry box
canvas1.create_window(660, 260, window=entry8)

label9 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='M') ## create 9st label box 
canvas1.create_window(430, 300, window=label9)
l4 = tk.Label(root, font=('microsoft yahei',10),text='(g/mol): ')
canvas1.create_window(525, 300, window=l4) 
entry9 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 9nd entry box
canvas1.create_window(660,300, window=entry9)

## The linkage between the physical properties of molecules and molecules is realized
## Sets four properties: 1-dia,2-pol,3-Dip,4-Qua
input1 = 0
input2 = 0
input3 = 0
input4 = 0
def run2():
    dic1 = {0: 'Dia', 1: 'Pol', 2: 'Dip',3: 'M'}
    c2 = dic1[cm1.current()]
    if (c2 == 'Dia'):                      
        entry6 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input1,justify='center')
        canvas1.create_window(660,180, window=entry6)              
    elif (c2 =='Pol'):
        entry7 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input2,justify='center') 
        canvas1.create_window(660,220, window=entry7)
    elif (c2 =='Dip'):
        entry8 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input3,justify='center') 
        canvas1.create_window(660, 260, window=entry8)                   
    elif (c2 =='M'):
        entry9 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input4,justify='center') 
        canvas1.create_window(660,300, window=entry9)
 
def calc(event):
    global input1
    global input2
    global input3
    global input4    
    dic = {0: 'C1', 1: 'C2', 2: 'C3',3: 'C4',4: 'C5',5: 'C6',6: 'C7'}
    c = dic[cm1.current()] 
    if (c == 'C1'):   # 根据选中的气体设置input变量的值，并更新GUI中的输入框内容
        # 设置氦气(C1)相关的属性值
        input1 = 3.758
        input2 = 25.93
        input3 = 0
        input4 = 16.05
        # 清空并更新GUI中的输入框(entry6至entry9)内容
        entry6.delete(0,"end") # 清空entry6的内容
        entry6.insert(0,"3.758") # 插入新的值
        entry7.delete(0,"end") # 清空entry7的内容
        entry7.insert(0,"25.93")  # 插入新的值
        entry8.delete(0,"end")
        entry8.insert(0,"0")
        entry9.delete(0,"end")
        entry9.insert(0,"16.05")#在这个GUI应用程序的代码中，虽然变量 input1, input2, input3, input4 已经被赋值，但这些只是程序后端的变量值。为了让这些值对用户可见，并让用户能够在界面上看到这些更新，需要将这些值显示在GUI的输入框中。这就是为什么需要进行清空和更新输入框的操作。
    elif (c == 'C2'):   
        input1 = 4.443
        input2 = 44.5
        input3 = 0
        input4 = 30.069
        entry6.delete(0,"end")
        entry6.insert(0,"4.443")
        entry7.delete(0,"end")
        entry7.insert(0,"44.5")
        entry8.delete(0,"end")
        entry8.insert(0,"0")
        entry9.delete(0,"end")
        entry9.insert(0,"30.069")
    elif (c == 'C3'):
        input1 = 4.3
        input2 = 62.9
        input3 = 0.084
        input4 = 44.11
        entry6.delete(0,"end")
        entry6.insert(0,"4.3")
        entry7.delete(0,"end")
        entry7.insert(0,"62.9")
        entry8.delete(0,"end")
        entry8.insert(0,"0.084")
        entry9.delete(0,"end")
        entry9.insert(0,"44.11")
    elif (c == 'C4'):
        input1 = 4.687
        input2 = 82
        input3 = 0.05
        input4 = 58.12
        entry6.delete(0,"end")
        entry6.insert(0,"4.687")
        entry7.delete(0,"end")
        entry7.insert(0,"82")
        entry8.delete(0,"end")
        entry8.insert(0,"0.05")
        entry9.delete(0,"end")
        entry9.insert(0,"58.12")
    elif (c == 'C5'):
        input1 = 4.5
        input2 = 99.9
        input3 = 0.13
        input4 = 72.15
        entry6.delete(0,"end")
        entry6.insert(0,"4.5")
        entry7.delete(0,"end")
        entry7.insert(0,"99.9")
        entry8.delete(0,"end")
        entry8.insert(0,"0.13")
        entry9.delete(0,"end")
        entry9.insert(0,"72.15")  
    elif (c == 'C6'):
        input1 = 4.3
        input2 = 119
        input3 = 0
        input4 = 86.18
        entry6.delete(0,"end")
        entry6.insert(0,"4.3")
        entry7.delete(0,"end")
        entry7.insert(0,"119")
        entry8.delete(0,"end")
        entry8.insert(0,"0")
        entry9.delete(0,"end")
        entry9.insert(0,"86.18") 
    elif (c == 'C7'):
        input1 = 4.3
        input2 = 136.1
        input3 = 0
        input4 = 100.21
        entry6.delete(0,"end")
        entry6.insert(0,"4.3")
        entry7.delete(0,"end")
        entry7.insert(0,"136.1")
        entry8.delete(0,"end")
        entry8.insert(0,"0")
        entry9.delete(0,"end")
        entry9.insert(0,"100.21") 

        
## Create a Drop-down box
var1 = tk.StringVar() ## Create a variable 
cm1 = ttk.Combobox(root, textvariable=var1,font=('microsoft yahei',10)) ## Create a drop-down menu
cm1["value"] = ("C1", "C2", "C3","C4","C5","C6","C7") ## The contents of a drop-down menu
canvas1.create_window(620,140, window=cm1)
cm1.bind('<<ComboboxSelected>>', calc) ## Binding 'calc' events
  
## Main interface for input of nine descriptor values (a single molecule diffusivity)
def values():       
    global New_φ #our 1st input variable    
    New_φ = float(entry1.get()) 
    
    global New_PLD #our 2nd input variable
    New_PLD = float(entry2.get()) 
    
    global New_LCD #our 2nd input variable
    New_LCD = float(entry3.get()) 
    
    global New_ρ #our 2nd input variable
    New_ρ = float(entry4.get()) 
    
    global New_VSA #our 2nd input variable
    New_VSA =float(entry5.get()) 
    
    global New_Dia #our 2nd input variable
    New_Dia = float(entry6.get()) 
    
    global New_Pol #our 2nd input variable
    New_Pol = float(entry7.get()) 
    
    global New_Dip #our 2nd input variable
    New_Dip = float(entry8.get())
    
    global New_M #our 2nd input variable
    New_M = float(entry9.get())

## LGBM Algorithm (The predictions of a single molecule diffusivity)   
    lgD = model.predict([[New_φ, New_PLD, New_LCD, New_ρ, New_VSA, New_Dia,
                          New_Pol, New_Dip, New_M]])    
    ## D transformation            
    D = pow(10,lgD)
    D1 = float(D)
    D2=format(D1,'.2E')
    D3= D2.split('E')  
    if (D3[1])[0] == "-":
        D3= D3[0]+" x 10^"+ D3[1].lstrip('0')
    else:
        D3=D3[0]+" x 10^"+ (D3[1])[1:].lstrip('0')    
    
    ## label of the predicted result
    Prediction_result  = (D3)   
    label_Prediction = tk.Label(root, font=('microsoft yahei',12),width=30,height=2,
                                text= Prediction_result)
    canvas1.create_window(420, 380, window=label_Prediction)
 
    ## D label
    lbo1=tk.Label(root, font=('microsoft yahei',12,"italic"),
                                text='D:')
    canvas1.create_window(330, 380, window=lbo1)
    
    ## unit label
    lbo2=tk.Label(root, font=('microsoft yahei',12),
                                text='(cm^3/s)')
    canvas1.create_window(540, 380, window=lbo2) 

## button to call the 'values' command above       
button1 = tk.Button (root,font=('microsoft yahei',10), text='Predicted D',command=values, bg='deeppink', fg='deeppink')
button1.configure(bg='coral') 
canvas1.create_window(130, 380, window=button1)

## Batch prediction of material diffusivity
label_Z1 = tk.Label(root,font=('microsoft yahei',12),text='Batch prediction of material diffusivity')
canvas1.create_window(415, 470, window=label_Z1)

## Open File
def open_file():
    filename = filedialog.askopenfilename(title='open exce')
    entry_filename.delete(0,"end")
    entry_filename.insert('insert', filename)
 
button_import = tk.Button(root, text="Import File",font=('microsoft yahei',10),command=open_file)
button_import.configure(bg='coral') 
canvas1.create_window(280, 520, window=button_import)
 
## Import File
entry_filename = tk.Entry(root,font=('microsoft yahei',10),width=30)
canvas1.create_window(520, 520, window=entry_filename)

## Output LGBM model prediction results
def print_file():

    ## get extract contents of entry
    a = entry_filename.get() 

    ## Load the dataset
    pred_data1=pd.read_excel(a)
    
    print("原始数据集大小：", pred_data1.shape)

    ## Divide the data set
    for column in pred_data1.select_dtypes(include=[np.number]).columns:
       pred_data1[column].fillna(pred_data1[column].mean(), inplace=True)


    
    
    print("移除NaN后的数据集大小：", pred_data1.shape)

    ## Divide the data set
    df = pd.DataFrame(pred_data1,columns=[ 'LCD', 'φ', 'VSA', 'PLD', 'ρ', 'Dia',
         'Pol', 'Dip', 'M','lgD'])
    X_pred = df[['LCD', 'φ', 'VSA', 'PLD', 'ρ', 'Dia',
         'Pol', 'Dip', 'M']].astype(float)

    ## Standardization
    transfer=StandardScaler()
    X_pred=transfer.fit_transform(X_pred)  
  
    ##model prediction
    Y_predict2 = model.predict(X_pred) 
    D = pow(10,Y_predict2) ## D transformation
    
    D = D / 6.5

    ## output result
    d1 = pd.DataFrame({'D_pred':D}) 
    newdata = pd.concat([pred_data1,d1],axis=1) 
    newdata.to_excel(r"Result\Batch_Predicted_D.xlsx")
    
    ## label_P (Prediction complete)
    label_P = tk.Label(root, font=('microsoft yahei',12),
                                text='Predicted results have default stored in:\nResult/Batch_Predicted_D.xlsx', bg='orange')
    canvas1.create_window(450, 600, window=label_P)    
    
## Prediction button
but_pre=tk.Button(root,font=('microsoft yahei',10)
             , text='Batch Predicted D', bg='deeppink', fg='deeppink', command=print_file)
canvas1.create_window(160, 600, window=but_pre)

root.mainloop() 
