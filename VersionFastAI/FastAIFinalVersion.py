import fastai, os, cv2 , PIL
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.gan import *

path_hr = 'drive/My Drive/Dataset/ds-medium-crap'
path_lr = 'drive/My Drive/Dataset/ds-medium'
import glob
import shutil
import os

src_dir = path_hr
dst_dir = path_lr
count=0
for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
    print(count)
    shutil.copy(jpgfile, dst_dir)
    count+=1
from PIL import Image, ImageDraw, ImageFont

def crappify(fn,i):
    dest = path_lr/fn.relative_to(path_hr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = PIL.Image.open(fn)
    targ_sz = resize_to(img, 256, use_min=True)
    img = img.resize(targ_sz).convert('L')
    img.save(dest)
    print("Crappified: " + str(count))
    count+=1
il = ImageItemList.from_folder(path_hr)
parallel(crappify, il.items)

bs,size = 24,160

path_hr = 'drive/My Drive/image_data'
path_lr = 'drive/My Drive/image_data_2'
arch = models.resnet34
src = ImageImageList.from_folder(path_lr).random_split_by_pct(0.2, seed=42)

def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_hr+"/"+x.name)
           .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
           .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data
    
data_gen = get_data(bs,size)
data_gen.show_batch(4)

wd = 1e-3
y_range = (-3.,3.)
loss_gen = MSELossFlat()
def create_gen_learner():
    return unet_learner(data_gen, arch, wd=wd, norm_type=NormType.Weight,
                         self_attention=True, y_range=y_range, loss_func=loss_gen)
                         
learn_gen = create_gen_learner()
learn_gen.fit_one_cycle(5, pct_start=0.8)
learn_gen.unfreeze()
learn_gen.fit_one_cycle(5, pct_start=0.8)
learn_gen.show_results(rows=10)
