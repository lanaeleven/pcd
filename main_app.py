from tkinter import *
# from tkinter.ttk import *
from tkinter.messagebox import showwarning
from PIL import ImageTk, Image
from tkinter import filedialog
import cv2
import skimage.feature as feature
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans

root = Tk()
root.title("GLCM and HSV")

blank_img = ImageTk.PhotoImage(Image.open('blank_icon.jpg').resize((150,150)))
path = None
path3 = None

ocean_fish_dataset = IntVar(value=1)

contrast_0 = StringVar()
contrast_45 = StringVar()
contrast_90 = StringVar()
contrast_135 = StringVar()
dissimilarity_0 = StringVar()
dissimilarity_45 = StringVar()
dissimilarity_90 = StringVar()
dissimilarity_135 = StringVar()
homogeneity_0 = StringVar()
homogeneity_45 = StringVar()
homogeneity_90 = StringVar()
homogeneity_135 = StringVar()
energy_0 = StringVar()
energy_45 = StringVar()
energy_90 = StringVar()
energy_135 = StringVar()
correlation_0 = StringVar()
correlation_45 = StringVar()
correlation_90 = StringVar()
correlation_135 = StringVar()
asm_0 = StringVar()
asm_45 = StringVar()
asm_90 = StringVar()
asm_135 = StringVar()

hue = StringVar()
saturation = StringVar()
value = StringVar()


content = Frame(root, bg="#edf2fb")

ori_img = Label(content, image=blank_img, bg="#edf2fb")
ori_img_label = Label(content, text="Original Image", bg="#edf2fb")
select_img_button = Button(content, text="Select Image", height=2, bg="#ccdbfd")
feature_extraction_button = Button(content, text="Feature Extraction", height=2, bg="#ccdbfd")
reset_button = Button(content, text="Reset", height=2, bg="#ccdbfd")
ocean_fish_dataset_checkbutton = Checkbutton(content, text = 'Image from ocean fish dataset', variable = ocean_fish_dataset, onvalue = 1, offvalue = 0)
dom_color_label = Label(content, text="Dominant Color for HSV:")
dom_color_entry = Entry(content)

grayscale_img = Label(content, image=blank_img)
grayscale_img_label = Label(content, text="Grayscale Image")
glcm_values_label = Label(content, text="GLCM Values", bg="#abc4ff", borderwidth=1, relief="solid")
contrast_label = Label(content, text="Contrast", bg="#abc4ff", borderwidth=1, relief="solid")
dissimilarity_label = Label(content, text="Dissimilarity", bg="#abc4ff", borderwidth=1, relief="solid")
homogeneity_label = Label(content, text="Homogeneity", bg="#abc4ff", borderwidth=1, relief="solid")
energy_label = Label(content, text="Energy", bg="#abc4ff", borderwidth=1, relief="solid")
correlation_label = Label(content, text="Correlation", bg="#abc4ff", borderwidth=1, relief="solid")
asm_label = Label(content, text="ASM", bg="#abc4ff", borderwidth=1, relief="solid")
angle_0_label = Label(content, text="0", width=30, bg="#abc4ff", borderwidth=1, relief="solid")
angle_45_label = Label(content, text="45", width=30, bg="#abc4ff", borderwidth=1, relief="solid")
angle_90_label = Label(content, text="90", width=30, bg="#abc4ff", borderwidth=1, relief="solid")
angle_135_label = Label(content, text="135", width=30, bg="#abc4ff", borderwidth=1, relief="solid")
contrast_0_label = Label(content, textvariable=contrast_0, bg="#c1d3fe", borderwidth=1, relief="solid")
contrast_45_label = Label(content, textvariable=contrast_45, bg="#c1d3fe", borderwidth=1, relief="solid")
contrast_90_label = Label(content, textvariable=contrast_90, bg="#c1d3fe", borderwidth=1, relief="solid")
contrast_135_label = Label(content, textvariable=contrast_135, bg="#c1d3fe", borderwidth=1, relief="solid")
dissimilarity_0_label = Label(content, textvariable=dissimilarity_0, bg="#d7e3fc", borderwidth=1, relief="solid")
dissimilarity_45_label = Label(content, textvariable=dissimilarity_45, bg="#d7e3fc", borderwidth=1, relief="solid")
dissimilarity_90_label = Label(content, textvariable=dissimilarity_90, bg="#d7e3fc", borderwidth=1, relief="solid")
dissimilarity_135_label = Label(content, textvariable=dissimilarity_135, bg="#d7e3fc", borderwidth=1, relief="solid")
homogeneity_0_label = Label(content, textvariable=homogeneity_0, bg="#c1d3fe", borderwidth=1, relief="solid")
homogeneity_45_label = Label(content, textvariable=homogeneity_45, bg="#c1d3fe", borderwidth=1, relief="solid")
homogeneity_90_label = Label(content, textvariable=homogeneity_90, bg="#c1d3fe", borderwidth=1, relief="solid")
homogeneity_135_label = Label(content, textvariable=homogeneity_135, bg="#c1d3fe", borderwidth=1, relief="solid")
energy_0_label = Label(content, textvariable=energy_0, bg="#d7e3fc", borderwidth=1, relief="solid")
energy_45_label = Label(content, textvariable=energy_45, bg="#d7e3fc", borderwidth=1, relief="solid")
energy_90_label = Label(content, textvariable=energy_90, bg="#d7e3fc", borderwidth=1, relief="solid")
energy_135_label = Label(content, textvariable=energy_135, bg="#d7e3fc", borderwidth=1, relief="solid")
correlation_0_label = Label(content, textvariable=correlation_0, bg="#c1d3fe", borderwidth=1, relief="solid")
correlation_45_label = Label(content, textvariable=correlation_45, bg="#c1d3fe", borderwidth=1, relief="solid")
correlation_90_label = Label(content, textvariable=correlation_90, bg="#c1d3fe", borderwidth=1, relief="solid")
correlation_135_label = Label(content, textvariable=correlation_135, bg="#c1d3fe", borderwidth=1, relief="solid")
asm_0_label = Label(content, textvariable=asm_0, bg="#d7e3fc", borderwidth=1, relief="solid")
asm_45_label = Label(content, textvariable=asm_45, bg="#d7e3fc", borderwidth=1, relief="solid")
asm_90_label = Label(content, textvariable=asm_90, bg="#d7e3fc", borderwidth=1, relief="solid")
asm_135_label = Label(content, textvariable=asm_135, bg="#d7e3fc", borderwidth=1, relief="solid")

hsv_img = Label(content, image=blank_img, bg="#edf2fb")
hsv_img_label = Label(content, text="HSV Image", bg="#edf2fb")
hue_label = Label(content, text="Hue", bg="#abc4ff", borderwidth=1, relief="solid")
saturation_label = Label(content, text="Saturation", bg="#abc4ff", borderwidth=1, relief="solid")
value_label = Label(content, text="Value", bg="#abc4ff", borderwidth=1, relief="solid")
hue_value = Label(content, textvariable=hue, bg="#c1d3fe", borderwidth=1, relief="solid")
saturation_value = Label(content, textvariable=saturation, bg="#c1d3fe", borderwidth=1, relief="solid")
value_value = Label(content, textvariable=value, bg="#c1d3fe", borderwidth=1, relief="solid")
export_data_button = Button(content, text="Export Data", height=2, bg="#ccdbfd")





def select_image():
    global path
    global path3
    path3 = None
    path = filedialog.askopenfilename()

    if ocean_fish_dataset.get() == 1:
        path2 = path.split('/')
        if path2[len(path2)-2] == "Black Sea Sprat":
            path2[len(path2)-2] = path2[len(path2)-2] + " GT"
            path3 = ""
            for x in path2:
                path3 = path3 +  x + "/"
            path3 = path3[:-1]
        else:
            showwarning(message="Image is not from ocean fish dataset")
            return
    
    img2 = ImageTk.PhotoImage(Image.open(path).resize((150,150)))
    ori_img.configure(image=img2)
    ori_img.image = img2
    print(ocean_fish_dataset)
    print(path3)


def feature_extraction():
    # gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    img0=cv2.imread(path) 
    if ocean_fish_dataset.get() == 1 and path3 != None:
        # segmentation
        img0t = cv2.cvtColor(img0,cv2.COLOR_BGRA2RGBA)
        mask_img1=cv2.imread(path3,cv2.IMREAD_GRAYSCALE)
        mask_img2=mask_img1/255
        img0t[:,:,3]=mask_img1
        img0p=img0.copy()
        img0p[:,:,0]=img0p[:,:,0]*mask_img2
        img0p[:,:,1]=img0p[:,:,1]*mask_img2
        img0p[:,:,2]=img0p[:,:,2]*mask_img2
        img0 = img0p
    
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
    image = hsv.reshape((hsv.shape[0] * hsv.shape[1], 3))
    clt = KMeans(n_clusters=2)
    labels = clt.fit_predict((image))
    label_counts = Counter(labels)
    dom_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    print(dom_color)
    hue.set(dom_color[0])
    saturation.set(dom_color[1])
    value.set(dom_color[2])

    # image_spot = img0
    # gray = cv2.cvtColor(image_spot, cv2.COLOR_BGR2GRAY)
    graycom = feature.graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
    contrast = feature.graycoprops(graycom, 'contrast')
    dissimilarity = feature.graycoprops(graycom, 'dissimilarity')
    homogeneity = feature.graycoprops(graycom, 'homogeneity')
    energy = feature.graycoprops(graycom, 'energy')
    correlation = feature.graycoprops(graycom, 'correlation')
    asm = feature.graycoprops(graycom, 'ASM')

    contrast_0.set(contrast[0][0])
    contrast_45.set(contrast[0][1])
    contrast_90.set(contrast[0][2])
    contrast_135.set(contrast[0][3])
    dissimilarity_0.set(dissimilarity[0][0])
    dissimilarity_45.set(dissimilarity[0][1])
    dissimilarity_90.set(dissimilarity[0][2])
    dissimilarity_135.set(dissimilarity[0][3])
    homogeneity_0.set(homogeneity[0][0])
    homogeneity_45.set(homogeneity[0][1])
    homogeneity_90.set(homogeneity[0][2])
    homogeneity_135.set(homogeneity[0][3])
    energy_0.set(energy[0][0])
    energy_45.set(energy[0][1])
    energy_90.set(energy[0][2])
    energy_135.set(energy[0][3])
    correlation_0.set(correlation[0][0])
    correlation_45.set(correlation[0][1])
    correlation_90.set(correlation[0][2])
    correlation_135.set(correlation[0][3])
    asm_0.set(asm[0][0])
    asm_45.set(asm[0][1])
    asm_90.set(asm[0][2])
    asm_135.set(asm[0][3])

    gray = Image.fromarray(gray).resize((150,150))
    hsv = Image.fromarray(hsv).resize((150,150))

    gray = ImageTk.PhotoImage(gray)
    hsv = ImageTk.PhotoImage(hsv)

    grayscale_img.configure(image=gray)
    grayscale_img.image = gray

    hsv_img.configure(image=hsv)
    hsv_img.image = hsv

def reset_image():
    path = None
    path3 = None
    ori_img.configure(image=blank_img)
    ori_img.image = blank_img
    grayscale_img.configure(image=blank_img)
    grayscale_img.image = blank_img
    hsv_img.configure(image=blank_img)
    hsv_img.image = blank_img

select_img_button.configure(command=select_image)
feature_extraction_button.configure(command=feature_extraction)
reset_button.configure(command=reset_image)

for child in content.winfo_children(): 
    child.grid_configure(sticky=(N,W,E,S))

# for child in content.winfo_children(): 
#     child.configure(bg="#abc4ff")

content.grid(column=0, row=0, ipadx=20, ipady=20)

ori_img.grid(column=1, row=0, rowspan=3, padx=(0,10))
ori_img_label.grid(column=1, row=3)
select_img_button.grid(column=2, row=0, sticky=(W,E))
feature_extraction_button.grid(column=2, row=1, sticky=(W,E))
reset_button.grid(column=2, row=2, sticky=(W,E))
ocean_fish_dataset_checkbutton.grid(column=3, row=0)
dom_color_label.grid(column=3, row=1, sticky=(E))
dom_color_entry.grid(column=4, row=1, sticky=(W))

grayscale_img.grid(column=1, row=4, rowspan=6, pady=(10, 0), padx=(0,10))
grayscale_img_label.grid(column=1, row=10)
glcm_values_label.grid(column=2, row=4, pady=(10, 0))
contrast_label.grid(column=2, row=5)
dissimilarity_label.grid(column=2, row=6)
homogeneity_label.grid(column=2, row=7)
energy_label.grid(column=2, row=8)
correlation_label.grid(column=2, row=9)
asm_label.grid(column=2, row=10)
angle_0_label.grid(column=3, row=4, pady=(10, 0))
angle_45_label.grid(column=4, row=4, pady=(10, 0))
angle_90_label.grid(column=5, row=4, pady=(10, 0))
angle_135_label.grid(column=6, row=4, pady=(10, 0))
contrast_0_label.grid(column=3, row=5)
contrast_45_label.grid(column=4, row=5)
contrast_90_label.grid(column=5, row=5)
contrast_135_label.grid(column=6, row=5)
dissimilarity_0_label.grid(column=3, row=6)
dissimilarity_45_label.grid(column=4, row=6)
dissimilarity_90_label.grid(column=5, row=6)
dissimilarity_135_label.grid(column=6, row=6)
homogeneity_0_label.grid(column=3, row=7)
homogeneity_45_label.grid(column=4, row=7)
homogeneity_90_label.grid(column=5, row=7)
homogeneity_135_label.grid(column=6, row=7)
energy_0_label.grid(column=3, row=8)
energy_45_label.grid(column=4, row=8)
energy_90_label.grid(column=5, row=8)
energy_135_label.grid(column=6, row=8)
correlation_0_label.grid(column=3, row=9)
correlation_45_label.grid(column=4, row=9)
correlation_90_label.grid(column=5, row=9)
correlation_135_label.grid(column=6, row=9)
asm_0_label.grid(column=3, row=10)
asm_45_label.grid(column=4, row=10)
asm_90_label.grid(column=5, row=10)
asm_135_label.grid(column=6, row=10)

hsv_img.grid(column=1, row=11, rowspan=6, pady=(20, 0), padx=(0,10))
hsv_img_label.grid(column=1, row=17)
hue_label.grid(column=2, row=11, pady=(20, 0))
saturation_label.grid(column=2, row=12)
value_label.grid(column=2, row=13)
hue_value.grid(column=3, row=11, pady=(20, 0))
saturation_value.grid(column=3, row=12)
value_value.grid(column=3, row=13)
export_data_button.grid(column=6, row=16)


root.mainloop()