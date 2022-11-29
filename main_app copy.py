from tkinter import *
from tkinter.ttk import *
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

# style = Style()
# style.configure("BW.TLabel", background="red")

frame_ori = LabelFrame(root, text="ori", style="BW.TLabel").grid(column=1, row=1)
frame_ori_desc = LabelFrame(root, text="ori_desc").grid(column=2, row=1)
frame_glcm = LabelFrame(root, text="glcm").grid(column=1, row=2)
frame_glcm_desc = LabelFrame(root, text="glcm_desc").grid(column=2, row=2)
frame_hsv = LabelFrame(root, text="hsv").grid(column=1, row=3)
frame_hsv_desc = LabelFrame(root, text="hsv_desc").grid(column=3, row=3)

ori_img = Label(frame_ori, image=blank_img)
ori_img_label = Label(frame_ori, text="Original Image")

grayscale_img = Label(frame_glcm, image=blank_img)
grayscale_img_label = Label(frame_glcm, text="Grayscale Image")

contrast_label = Label(frame_glcm_desc, text="Contrast")
dissimilarity_label = Label(frame_glcm_desc, text="Dissimilarity")
homogeneity_label = Label(frame_glcm_desc, text="Homogeneity")
energy_label = Label(frame_glcm_desc, text="Energy")
correlation_label = Label(frame_glcm_desc, text="Correlation")
asm_label = Label(frame_glcm_desc, text="ASM")
angle_0_label = Label(frame_glcm_desc, text="0", width=30)
angle_45_label = Label(frame_glcm_desc, text="45", width=30)
angle_90_label = Label(frame_glcm_desc, text="90", width=30)
angle_135_label = Label(frame_glcm_desc, text="135", width=30)
contrast_0_label = Label(frame_glcm_desc, textvariable=contrast_0)
contrast_45_label = Label(frame_glcm_desc, textvariable=contrast_45)
contrast_90_label = Label(frame_glcm_desc, textvariable=contrast_90)
contrast_135_label = Label(frame_glcm_desc, textvariable=contrast_135)
dissimilarity_0_label = Label(frame_glcm_desc, textvariable=dissimilarity_0)
dissimilarity_45_label = Label(frame_glcm_desc, textvariable=dissimilarity_45)
dissimilarity_90_label = Label(frame_glcm_desc, textvariable=dissimilarity_90)
dissimilarity_135_label = Label(frame_glcm_desc, textvariable=dissimilarity_135)
homogeneity_0_label = Label(frame_glcm_desc, textvariable=homogeneity_0)
homogeneity_45_label = Label(frame_glcm_desc, textvariable=homogeneity_45)
homogeneity_90_label = Label(frame_glcm_desc, textvariable=homogeneity_90)
homogeneity_135_label = Label(frame_glcm_desc, textvariable=homogeneity_135)
energy_0_label = Label(frame_glcm_desc, textvariable=energy_0)
energy_45_label = Label(frame_glcm_desc, textvariable=energy_45)
energy_90_label = Label(frame_glcm_desc, textvariable=energy_90)
energy_135_label = Label(frame_glcm_desc, textvariable=energy_135)
correlation_0_label = Label(frame_glcm_desc, textvariable=correlation_0)
correlation_45_label = Label(frame_glcm_desc, textvariable=correlation_45)
correlation_90_label = Label(frame_glcm_desc, textvariable=correlation_90)
correlation_135_label = Label(frame_glcm_desc, textvariable=correlation_135)
asm_0_label = Label(frame_glcm_desc, textvariable=asm_0)
asm_45_label = Label(frame_glcm_desc, textvariable=asm_45)
asm_90_label = Label(frame_glcm_desc, textvariable=asm_90)
asm_135_label = Label(frame_glcm_desc, textvariable=asm_135)

hsv_img = Label(frame_hsv, image=blank_img)
hsv_img_label = Label(frame_hsv, text="HSV Image")






def select_image():
    global path
    global path3
    path = filedialog.askopenfilename()
    img2 = ImageTk.PhotoImage(Image.open(path).resize((150,150)))
    ori_img.configure(image=img2)
    ori_img.image = img2
    print(path)
    path2 = path.split('/')
    path2[2] = path2[2] + " GT"
    path3 = ""
    for x in path2:
        path3 = path3 +  x + "/"
    path3 = path3[:-1]

def feature_extraction():
    # gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

    img0=cv2.imread(path) 
    img0t = cv2.cvtColor(img0,cv2.COLOR_BGRA2RGBA)
    mask_img1=cv2.imread(path3,cv2.IMREAD_GRAYSCALE)
    mask_img2=mask_img1/255
    img0t[:,:,3]=mask_img1
    img0p=img0.copy()
    img0p[:,:,0]=img0p[:,:,0]*mask_img2
    img0p[:,:,1]=img0p[:,:,1]*mask_img2
    img0p[:,:,2]=img0p[:,:,2]*mask_img2
    
    gray = cv2.cvtColor(img0p, cv2.COLOR_BGR2GRAY)
    # src = cv2.imread(path, 1)
    # tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # _,mask = cv2.threshold(tmp, 127, 255, cv2.THRESH_BINARY_INV)
    # mask = cv2.dilate(mask.copy(), None, iterations=10)
    # mask = cv2.erode(mask.copy(), None, iterations=10)
    # b,g,r = cv2.split(src)
    # rgba = [b,g,r,mask]
    # dst = cv2.merge(rgba, 4)

    # contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # selected = max(contours,key=cv2.contourArea)
    # x,y,w,h = cv2.boundingRect(selected)
    # cropped = dst[y:y+h, x:x+w]
    # mask = mask[y:y+h, x:x+w]
    # gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # hsv = cv2.imread(path)
    hsv = cv2.cvtColor(img0p, cv2.COLOR_BGR2HSV)
    image = hsv.reshape((hsv.shape[0] * hsv.shape[1], 3))
    clt = KMeans(n_clusters=2)
    labels = clt.fit_predict((image))
    label_counts = Counter(labels)
    dom_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    print(dom_color)


    gray = Image.fromarray(gray).resize((150,150))
    hsv = Image.fromarray(hsv).resize((150,150))

    gray = ImageTk.PhotoImage(gray)
    hsv = ImageTk.PhotoImage(hsv)

    grayscale_img.configure(image=gray)
    grayscale_img.image = gray

    hsv_img.configure(image=hsv)
    hsv_img.image = hsv

    image_spot = cv2.imread(path)
    gray = cv2.cvtColor(image_spot, cv2.COLOR_BGR2GRAY)
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

select_img_button = Button(frame_ori_desc, text="Select Image", command=select_image)
feature_extraction_button = Button(frame_ori_desc, text="Feature Extraction", command=feature_extraction)

# content.columnconfigure(1, weight=3)
# content.columnconfigure(2, weight=2)
# content.columnconfigure(3, weight=4)
# content.columnconfigure(4, weight=4)
# content.columnconfigure(5, weight=4)
# content.columnconfigure(6, weight=4)

content.grid(column=0, row=0, sticky=(N, W, E, S))
ori_img.grid(column=1, row=0, rowspan=3)
grayscale_img.grid(column=1, row=4, rowspan=5)
hsv_img.grid(column=1, row=11, rowspan=3, padx=20, pady=20)
ori_img_label.grid(column=1, row=3)
grayscale_img_label.grid(column=1, row=9)
hsv_img_label.grid(column=1, row=14)
select_img_button.grid(column=2, row=1)
feature_extraction_button.grid(column=2, row=2)
contrast_label.grid(column=2, row=5)
dissimilarity_label.grid(column=2, row=6)
homogeneity_label.grid(column=2, row=7)
energy_label.grid(column=2, row=8)
correlation_label.grid(column=2, row=9)
asm_label.grid(column=2, row=10)
angle_0_label.grid(column=3, row=4)
angle_45_label.grid(column=4, row=4)
angle_90_label.grid(column=5, row=4)
angle_135_label.grid(column=6, row=4)
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


# for child in content.winfo_children(): 
#     child.grid_configure(padx=20, pady=20)


# content = ttk.Frame(root)
# frame = ttk.Frame(content, borderwidth=5, relief="ridge", width=200, height=100)
# namelbl = ttk.Label(content, text="Name")
# name = ttk.Entry(content)

# onevar = BooleanVar(value=True)
# twovar = BooleanVar(value=False)
# threevar = BooleanVar(value=True)

# one = ttk.Checkbutton(content, text="One", variable=onevar, onvalue=True)
# two = ttk.Checkbutton(content, text="Two", variable=twovar, onvalue=True)
# three = ttk.Checkbutton(content, text="Three", variable=threevar, onvalue=True)
# ok = ttk.Button(content, text="Okay")
# cancel = ttk.Button(content, text="Cancel")

# content.grid(column=0, row=0)
# frame.grid(column=0, row=0, columnspan=3, rowspan=2)
# namelbl.grid(column=3, row=0, columnspan=2)
# name.grid(column=3, row=1, columnspan=2)
# one.grid(column=0, row=3)
# two.grid(column=1, row=3)
# three.grid(column=2, row=3)
# ok.grid(column=3, row=3)
# cancel.grid(column=4, row=3)

root.mainloop()