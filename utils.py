
import cv2
import os
import numpy as np

def show(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def file_path(folder, filename, region, ext='.jpg'):
    filename = filename.replace(ext, '')
    filename = "{}_{}{}".format(filename, region, ext)
    path = os.path.join(folder, filename)
    return path


def write_img_region(path, region, img):
    '''
    Saves a cropped area of the image with the area of the region

    :param folder:
    :param region:
    :param img:
    :param ext:
    :return:
    '''
    sliced_img = img[region.slice2, region.slice1]
    if cv2.imwrite(path, sliced_img):
        return sliced_img


# class Region(object):
#     def __init__(self, x, y, h, w):
#         self.x = x
#         self.y = y
#         self.h = h
#         self.w = w

class Region(object):
    def __init__(self, slice1, slice2, mask, img=None):
        self.img = img
        self.mask = mask
        self.slice1 = slice1
        self.slice2 = slice2
        self._overlap = None

    @property
    def overlap(self):
        if self._overlap is not None:
            return self._overlap
        window = self.mask[self.slice2, self.slice1]
        self._overlap = window.sum() / np.prod(window.shape).astype('float')
        return self._overlap

    def cropped_img(self):
        return self.img[self.slice2, self.slice1]

    def __repr__(self):
        return "{self.__class__.__name__}({self.slice1}, {self.slice2})".format(self=self)

    def __str__(self):
        return "region_{}_{}_{}_{}".format(self.slice1.start, self.slice1.stop, self.slice2.start, self.slice2.stop )


def line2region(line):
    vals = line.split(',')
    return tuple(map(int, vals))

def file2regions(fileobj):
    regions = []
    for line in fileobj:
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            continue
        regions.append(line2region(line))
    return regions


class SliceGenerator(object):
    ''

    parse_line = line2region

    def __init__(self, image_filename, region_filename, window=None, stride=None, start=None):
        self.window_y, self.window_x = window
        self.pos_x, self.pos_y = start
        self.stride = stride
        self.image_filename = image_filename
        self.region_filename = region_filename
        self.img = cv2.imread(self.image_filename, cv2.IMREAD_GRAYSCALE)
        # self.img = cv2.bitwise_not(self.img)
        self.img2 = np.where(self.img > self.img.mean(), 255, 0)
        self.class_img = np.zeros(self.img.shape)

        with open(self.region_filename, 'r') as f:
            self.regions = file2regions(f)

        for region in self.regions:
            mask(self.class_img, *region)
            x, y, w, h = region
            cv2.rectangle(self.img, (x, y), (x+w, y+h), (255,255,255), 3)

        cv2.imwrite('temp/image_regions.jpg', self.img)
        cv2.imwrite('temp/image2.jpg', self.img2)
        cv2.imwrite('temp/image_mask.jpg', np.where(self.class_img > 0, 255, 0))

        print self

    def __str__(self):
        msg = "{self.region_filename}\n" \
            "{self.img.shape}"

        return msg.format(self=self)

    # def current_dim(self):
    #     '''
    #     x, y, w, h
    #     '''
    #     x = self.pos_x
    #     y = self.pos_y
    #     w = self.window_x
    #     h = self.window_y
    #     return x, y, w, h

    def current_slice(self):
        x, y, w, h = self.pos_x, self.pos_y, self.window_y, self.window_x
        return slice(x, x+w), slice(y, y+h)

    def iter_windows(self):
        # Remember that numpy is in rows, cols
        # Zig zag across the image
        #               x
        #   start 1: -------->
        #    y    2: -------->
        #         n: -------->
        while True:
            y_end, x_end = self.img.shape
            x_slice, y_slice = self.current_slice()
            if x_slice.stop <= x_end and y_slice.stop <= y_end:
                yield x_slice, y_slice
                self.pos_x += self.stride
            if y_slice.stop > y_end:
                raise StopIteration
            if x_slice.stop > x_end:
                self.pos_x = 0
                self.pos_y += self.stride





class DirectorySliceGenerator(object):

    def __init__(self, directory, slice_dim):
        self.directory = directory
        self.slice_dim = slice_dim
        self.selected_regions = []

    def images(self):
        'Finds all images for which there is a text fie'
        self.files_base = []
        for f in os.listdir(self.directory):
            base, ext = os.path.splitext(f)
            if ext.lower() == '.txt':
                self.files_base.append(base)

    def load_image(self, text_filename):
        self.text_filename = text_filename
        base, ext = os.path.splitext(self.text_filename)
        for filename in os.listdir(self.directory):
            if filename.startswith(base) and filename.lower().endswith('jpg'):
                self.image_filename = filename
        else:
            self.image_filename = None


    def gen_slices(self):
        ''

    def x(self, filename):
        self.img = img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        self.class_img = np.zeros(img.shape)

    def make_regions(self):
        with open(self.text_filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('#'):
                    continue



    def make_mask(self):
        for region in self.selected_regions:
            mask(self.class_img, region)

        for line in self:
            ''


def mask(img, x, y, w, h):
    ''
    img[y:y+h, x:x+w] = 1


if __name__ == '__main__':
    slice_gen = SliceGenerator(
        image_filename='images/kic_scanned_export_1174.jpg',
        region_filename='images/kic_scanned_export_1174.txt',
        window=(500, 100),
        stride=50,
        start=(0,0)
    )

    S = []
    gen = slice_gen.iter_windows()
    # show(slice_gen.class_img)
    for i, slices in enumerate(gen):
        s1, s2 = slices
        r = Region(s1, s2, mask=slice_gen.class_img, img=slice_gen.img)
        S.append(r)
        if r.overlap > .7:
            # saved_img = write_img_region(path=file_path('output', 'img', r),
            #                  region=r,
            #                  img=slice_gen.img)
            cls = '1'
        else:
            cls = '0'
        saved_img = cv2.imwrite(file_path('output', cls + '_img', r), r.cropped_img())
        assert saved_img



    # (112, 450, 933, 571)

    print len(S)
    print 'done'

