from PIL import Image
import glob


def resize_file(in_file, out_file, size):
    with Image.open(in_file) as fd:
        new_width, new_height = size
        fd = fd.resize((new_width, new_height), Image.ANTIALIAS)
    fd.save(out_file)
    fd.close()


for filename in glob.glob('/Users/rookie/antiSpider/nums/*.png'):
# for filename in glob.glob('/Users/rookie/antiSpider/testResize/*.png'):
    resize_file(filename, filename, (80, 40))
