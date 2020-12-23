try:
    import os
    import json
    import glob
    import argparse

    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter

    from PIL import Image

    import matplotlib.pyplot as plt

    import phase4.plot as plot

except ImportError:
    print("Need to fix the installation")
    raise


def high_pass_filter(img):
    highpass_filter = np.array([[-1 / 9, -1 / 9, -1 / 9],
                                [-1 / 9, 8 / 9, -1 / 9],
                                [-1 / 9, -1 / 9, -1 / 9]])
    return sg.convolve2d(img.T, highpass_filter, boundary='symm', mode='same')


def filter_by_color(img, color):
    return high_pass_filter(img[:, :, color])


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    highpass_red_filter = filter_by_color(c_image, 0)
    max_red_filter = maximum_filter(highpass_red_filter, 20)
    red_candidates = np.array([[i, j] for i in range(0, len(max_red_filter)) for j in range(0, len(max_red_filter[0])) if max_red_filter[i][j] == highpass_red_filter[i][j] and max_red_filter[i][j] > 30])
    x_red, y_red = [rc[0] for rc in red_candidates], [rc[1] for rc in red_candidates]

    highpass_green_filter = filter_by_color(c_image, 1)
    max_green_filter = maximum_filter(highpass_green_filter, 20)
    green_candidates = np.array([[i, j] for i in range(0, len(max_green_filter)) for j in range(0, len(max_green_filter[0])) if max_green_filter[i][j] == highpass_green_filter[i][j] and max_green_filter[i][j] > 30])
    x_green, y_green = [gc[0] for gc in green_candidates], [gc[1] for gc in green_candidates]
    return x_red, y_red, x_green, y_green


def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    open_image = Image.open(image_path)
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    show_image_and_gt(image, objects, fig_num)
    red_x, red_y, green_x, green_y = find_tfl_lights(image, open_image=open_image)
    # red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def main(argv=None):
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = '../../data'
    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)
    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


def find_lights(img_path, fig, title):
    image = np.array(Image.open(img_path))
    red_x, red_y, green_x, green_y = find_tfl_lights(image)
    candidates = [(x, y) for x, y in zip(red_x, red_y)]
    candidates += [(x, y) for x, y in zip(green_x, green_y)]
    auxiliary = ["red"] * len(red_x) + ["green"] * len(green_x)

    plot.mark_tfl(image, np.array(candidates), fig, len(red_x), title)
    # show_image_and_gt(np.array(Image.open(img_path)), None, None)
    # plt.plot(green_x, green_y, 'ro', color='g', markersize=4)
    # plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    # plt.show(block=True)
    return {"candidates": candidates, "auxiliary": auxiliary}

