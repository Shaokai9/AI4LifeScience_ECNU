import os

path = os.getcwd()
train2017_path = os.path.join(path,'./coco/images/train2017')

print(os.path.abspath(train2017_path))
hazy_files = os.listdir(train2017_path)
with open('./coco/train2017.txt','w+',encoding='utf-8') as f:
    for index,name in enumerate(hazy_files):
        # print(index,name)
        # mpath = os.getcwd()
        # x = Image.open(os.path.join(mpath,'train','1',name))
        # print(x)
        # break
        # txtline = str(index) + '\t' + os.path.join(mpath,'train','2',name)+ '    ' + os.path.join(mpath,'train','1',name) +'\n'
        # txtline = str(index) + '\t' + '/datasets/train/2/' +name + '    ' + '/datasets/train/1/'+ name + '\n'
        txtline = './images/train2017/'+name+'\n'
        f.write(txtline)


# clear_files = os.listdir(clear_path)
# print(hazy_files)
# print(clear_files)