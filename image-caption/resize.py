# coding: utf-8

from __future__ import print_function

from PIL import Image
import os


def resize_image(image, size):
    return image.resize(size, Image.ANTIALIAS)


def resize_images(image_dir, output_dir, size):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)

    for i, image in enumerate(images):
        if os.path.exists(os.path.join(output_dir, image)):
            continue
        with Image.open(os.path.join(image_dir, image)) as img:
            img = resize_image(img, size)
            img.save(os.path.join(output_dir, image), img.format)
        if i % 100 == 0:
            print('[%d/%d] Resized the images and saved into %s.'
                  % (i, num_images, output_dir))


def main():
    splits = ['train', 'val']
    for split in splits:
        image_dir = './data/%s2014/' % split
        output_dir = './data/%s2014resized/' % split
        resize_images(image_dir, output_dir, (256, 256))


if __name__ == '__main__':
    main()
