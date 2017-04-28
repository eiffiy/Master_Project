from PIL import Image
import os
path = "/Users/eiffiy/Desktop/jaffe"
files= os.listdir(path)
img_list = []
label_list = []
label = 0

for file in files:
    if not os.path.isdir(file):
        if file[3] is 'A':
            label = 1
        if file[3] is 'D':
            label = 2
        if file[3] is 'F':
            label = 3
        if file[3] is 'H':
            label = 4
        if file[3] is 'N':
            label = 5
        if file[3] is 'S':
            label = 6
        im = Image.open(path+"/"+file,"r")
        im.thumbnail((28,28));
        x = list(im.getdata())
        img_list.append(x)
        label_list.append(label)

data = [img_list, label_list]

print len(img_list[0])
print img_list[0]
print len(label_list)
