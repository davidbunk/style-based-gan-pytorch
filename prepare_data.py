import argparse
from io import BytesIO
import multiprocessing
from functools import partial

from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn


def resize_and_convert(img, quality=100):
    #img = trans_fn.resize(img, size)
    #img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format='jpeg', quality=quality)
    val = buffer.getvalue()

    return val


# def resize_multiple(img, sizes=(8, 16, 32, 64, 128, 256, 512, 1024), quality=100):
#     imgs = []
#
#     for size in sizes:
#         imgs.append(resize_and_convert(img, size, quality))
#
#     return imgs


def resize_worker(img_file, sizes):
    i, file = img_file
    img = Image.open(file)
    img = img.convert('L')
    size, _ = img.size
    out = resize_and_convert(img)

    return i, out, size


def prepare(transaction, dataset, n_worker, sizes=(8, 16, 32, 64, 128, 256, 512, 1024)):
    resize_fn = partial(resize_worker, sizes=sizes)

    files = sorted(dataset.imgs, key=lambda x: x[0])
    files = [(i, file) for i, (file, label) in enumerate(files)]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, img, size in tqdm(pool.imap_unordered(resize_fn, files)):
            #for size, img in zip(sizes, imgs):
            key = f'{size}-{str(i).zfill(5)}'.encode('utf-8')
            txn.put(key, img)

            total += 1

        txn.put(f'length-{size}'.encode('utf-8'), str(total).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--path', type=str)

    args = parser.parse_args()

    # temporary
    imgset_02 = datasets.ImageFolder('/scratch/guirao/datasets/Tubuli_dataset/nohighres_migseg_tifs/r02')
    imgset_03 = datasets.ImageFolder('/scratch/guirao/datasets/Tubuli_dataset/nohighres_migseg_tifs/r03')
    imgset_04 = datasets.ImageFolder('/scratch/guirao/datasets/Tubuli_dataset/nohighres_migseg_tifs/r04')
    imgset_05 = datasets.ImageFolder('/scratch/guirao/datasets/Tubuli_dataset/nohighres_migseg_tifs/r05')
    imgset_06 = datasets.ImageFolder('/scratch/guirao/datasets/Tubuli_dataset/nohighres_migseg_tifs/r06')
    imgset_07 = datasets.ImageFolder('/scratch/guirao/datasets/Tubuli_dataset/nohighres_migseg_tifs/r07')
    imgset_08 = datasets.ImageFolder('/scratch/guirao/datasets/Tubuli_dataset/nohighres_migseg_tifs/r08')
    imgset_09 = datasets.ImageFolder('/scratch/guirao/datasets/Tubuli_dataset/nohighres_migseg_tifs/r09')

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        with env.begin(write=True) as txn:
            prepare(txn, imgset_02, args.n_worker)
            prepare(txn, imgset_03, args.n_worker)
            prepare(txn, imgset_04, args.n_worker)
            prepare(txn, imgset_05, args.n_worker)
            prepare(txn, imgset_06, args.n_worker)
            prepare(txn, imgset_07, args.n_worker)
            prepare(txn, imgset_08, args.n_worker)
            prepare(txn, imgset_09, args.n_worker)
