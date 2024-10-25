import cv2
import os
from metrics import *
from tqdm import tqdm

eval_funcs = {
    "AG": ag,
    "CE": cross_entropy,
    "EI": edge_intensity,
    "EN": entropy,
    "MI": mutinf,
    "MSE": mse,
    "PSNR": psnr,
    "SD": sd,
    "SF": sf,
    "SSIM": ssim,
    "Qabf": qabf,
    "Qcb": qcb,
    "Qcv": qcv,
    "VIF": vif,
}


def main():
    path_fusimgs, path_irimgs, path_viimgs = [], [], []
    fus_root = "/home/tsy/yh/Diff-IF/experiments/VTUAV50/results"
    vi_root = "/home/tsy/yh/data/VTUAV50/RGB"
    ir_root = "/home/tsy/yh/data/VTUAV50/IR"
    metrics = eval_funcs.keys()

    img_list = os.listdir(fus_root)
    for img in img_list:
        if "\n" in img:
            img = img[:-1]
        if ".jpg" not in img and ".png" not in img:
            continue
        path_fusimgs.append(os.path.join(fus_root, img))
        path_irimgs.append(os.path.join(ir_root, img))
        path_viimgs.append(os.path.join(vi_root, img))

    if len(path_viimgs) != len(path_irimgs):
        print("The number of vi_imgs and ir_imgs are different!")

    res = {}
    for key in metrics:
        res[key] = [None] * len(path_fusimgs)

    pbar = iter(tqdm(range(len(path_fusimgs))))
    for i in range(len(path_fusimgs)):
        next(pbar)
        print("Now caculate the {}th img".format(i + 1))

        img_fus = cv2.imread(path_fusimgs[i], 0)
        img_vi = cv2.imread(path_viimgs[i], 0)
        img_ir = cv2.imread(path_irimgs[i], 0)
        max_h = img_fus.shape[0]
        max_w = img_fus.shape[1]
        img_fus = cv2.resize(img_fus, (max_w, max_h))
        img_vi = cv2.resize(img_vi, (max_w, max_h))
        img_ir = cv2.resize(img_ir, (max_w, max_h))

        for metric in metrics:
            try:
                res[metric][i] = eval_funcs[metric](img_fus)
            except:
                res[metric][i] = eval_funcs[metric](img_fus, img_vi, img_ir)

    N = len(path_fusimgs)
    for k, v in res.items():
        print(f"{k}: {sum(v)/N}")


if __name__ == "__main__":
    main()
