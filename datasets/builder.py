



import torch.utils.data as data
import os
import torch
import logging
import re
from PIL import Image



_logger = logging.getLogger(__name__)

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg')
_ERROR_RETRY = 50



def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            parser=None,
            class_map='',
            load_bytes=False,
            transform=None,
            **kwargs,
    ):
        if parser is None or isinstance(parser, str):
                assert os.path.exists(root)
                # default fallback path (backwards compat), use image tar if root is a .tar file, otherwise image folder
                # FIXME support split here, in parser?
                parser = ParserImageFolder(root, class_map=class_map,**kwargs)
        self.parser = parser
        self.load_bytes = load_bytes
        self.transform = transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.parser[index]
        try:
            img = img.read() if self.load_bytes else Image.open(img).convert('RGB')
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.parser.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.parser))
            else:
                raise e
        self._consecutive_errors = 0
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.tensor(-1, dtype=torch.long)
        return img, target

    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class ParserImageFolder(object):

    def __init__(
            self,
            root,
            class_ratio=1,
            sample_ratio=1,
            class_map=''):
        super().__init__()

        self.root = root
        class_to_idx = None
        self.class_ratio = class_ratio 
        self.sample_ratio = sample_ratio
        if class_map:
            class_to_idx = self.load_class_map(class_map, root)
        self.samples, self.class_to_idx = self.find_images_and_targets(root, class_to_idx=class_to_idx)
        if len(self.samples) == 0:
            raise RuntimeError(
                f'Found 0 images in subfolders of {root}. Supported image extensions are {", ".join(IMG_EXTENSIONS)}')

    def __getitem__(self, index):
        path, target = self.samples[index]
        return open(path, 'rb'), target

    def __len__(self):
        return len(self.samples)

    def  load_class_map(self, filename, root=''):
        class_map_path = filename
        if not os.path.exists(class_map_path):
            class_map_path = os.path.join(root, filename)
            assert os.path.exists(class_map_path), 'Cannot locate specified class map file (%s)' % filename
        class_map_ext = os.path.splitext(filename)[-1].lower()
        if class_map_ext == '.txt':
            with open(class_map_path) as f:
                class_to_idx = {v.strip(): k for k, v in enumerate(f)}
        else:
            assert False, 'Unsupported class map extension'
        return class_to_idx

    def find_images_and_targets(self,folder, class_to_idx=None, leaf_name_only=True, sort=True):
        types=IMG_EXTENSIONS
        labels = []
        filenames = []
        data_size = self.class_ratio
        ratio = self.sample_ratio

        for cnt,(root, subdirs, files) in enumerate(os.walk(folder, topdown=False, followlinks=True)):
            rel_path = os.path.relpath(root, folder) if (root != folder) else ''
            label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
            file_leng= len(files) * ratio
            for idx, f in enumerate(files):
                base, ext = os.path.splitext(f)
                if ext.lower() in types:
                    filenames.append(os.path.join(root, f))
                    labels.append(label)
                if idx >= file_leng -1 :
                    break
            if cnt >= 1000*data_size-1:
                print("dataset size is {}".format(cnt+1) )
                break
        if class_to_idx is None:
            # building class index
            unique_labels = set(labels)
            sorted_labels = list(sorted(unique_labels, key=natural_key))
            class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
        images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx]
        if sort:
            images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
        return images_and_targets, class_to_idx


    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename


def _search_split(root, split):
    # look for sub-folder with name of split in root and use that if it exists
    split_name = split.split('[')[0]
    try_root = os.path.join(root, split_name)
    if os.path.exists(try_root):
        return try_root
    if split_name == 'validation':
        try_root = os.path.join(root, 'val')
        if os.path.exists(try_root):
            return try_root
    return root

def create_dataset(name, root, split='validation', search_split=True, is_training=False, batch_size=None, **kwargs):
    name = name.lower()

    # FIXME support more advance split cfg for ImageFolder/Tar datasets in the future
    kwargs.pop('repeats', 0)  # FIXME currently only Iterable dataset support the repeat multiplier
    if search_split and os.path.isdir(root):
        root = _search_split(root, split)

    print(root)
    ds = ImageDataset(root, parser=name, **kwargs)
    return ds