import re
from os import walk
from os.path import join
import cv2

# CROP_SIZE = 224
CROP_SIZE = 384
RESIZE_FACTOR = 5


def get_list_of_images(dataset_path):
    data = []
    for (dirpath, dirnames, filenames) in walk(dataset_path):
        data.extend(filenames)
        break
    regex_jpg = re.compile(".jpg")
    regex_png = re.compile(".png")
    return [i for i in data if regex_jpg.search(i) or regex_png.search(i)]


if __name__ == "__main__":
    dataset_path = "E:/AiDatasets/Licenta/DIVerse2K_1/"
    new_dataset_path = "E:/AiDatasets/Licenta/SuperResolutionTesting/"
    images = get_list_of_images(dataset_path=dataset_path)

    for index, filename in enumerate(images):
        print("{}/{}".format(index + 1, len(images)))

        img = cv2.imread(join(dataset_path, filename))
        h, w, _ = img.shape

        for k_h in range((h // CROP_SIZE) - 1):
            for k_w in range((w // CROP_SIZE) - 1):


                crop = img[k_h*CROP_SIZE:(k_h + 1)*CROP_SIZE, k_w*CROP_SIZE:(k_w + 1)*CROP_SIZE]

                cv2.imwrite(join(new_dataset_path, "{}_{}_{}.png".format(index, k_h, k_w)), crop)

                # lr_crop_small = cv2.resize(crop, (CROP_SIZE//RESIZE_FACTOR, CROP_SIZE//RESIZE_FACTOR))
                # lr_crop_regular = cv2.resize(lr_crop_small, (CROP_SIZE, CROP_SIZE))
                # cv2.imwrite(join(new_dataset_path, "highres","{}.jpg".format(index)), crop)
                # cv2.imwrite(join(new_dataset_path, "lowres","{}.jpg".format(index)), lr_crop_regular)


