from datasets import load_dataset
import matplotlib.pyplot as plt

ds = load_dataset("facebook/winoground")

#print(ds)

sample = ds["test"][0]
# print(sample)

print(sample["caption_0"])
print(sample["caption_1"])

img0 = sample["image_0"]
img1 = sample["image_1"]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(img0)
axes[0].set_title("Image 0")

axes[1].imshow(img1)
axes[1].set_title("Image 1")


plt.show()