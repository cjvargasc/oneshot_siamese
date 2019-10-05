import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from selectivesearch import selectiveSearch
from skimage import img_as_float

test_im = cv2.imread("/home/mmv/Documents/camilo/datasets/openlogo/JPEGImages/3m4.jpg")
test_im = cv2.resize(test_im, None, fx=0.5, fy=0.5)
test_im = test_im[:, :, ::-1]
test_im = img_as_float(test_im)

img_lbl, regions = selectiveSearch.selective_search(
        test_im, scale=80.0, sigma=0.8, min_size=50) # scale= 70 or 60 sigma= 0.2 or 0.3 respectively

candidates = set()
for r in regions:
    # excluding same rectangle (with different segments)
    if r['rect'] in candidates:
        continue
    # excluding regions smaller than 2000 pixels
    if r['size'] < 300:
        continue
    # distorted rects
    x, y, w, h = r['rect']
    if h == 0 or w == 0:
        continue
    if w / h > 1.2 or h / w > 1.2:
        continue
    candidates.add(r['rect'])

# draw rectangles on the original image
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(test_im)
for x, y, w, h in candidates:
    #print(x, y, w, h)
    rect = mpatches.Rectangle(
        (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)

plt.show()