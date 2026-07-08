import os, shutil, xml.etree.ElementTree as ET
from pathlib import Path

# --- paths: change this to your VOC root ---
VOC_ROOT = r"/mnt/scratch2/users/40464858/VOC_dataset/VOCdevkit"  # contains VOC2007 and VOC2012
OUT_ROOT = r"/mnt/scratch2/users/40464858/VOC_dataset/voc_yolo"   # new YOLO-style dataset
SPLITS = {
    "train": [("VOC2007", "trainval"), ("VOC2012", "trainval")],  # 07+12 trainval as train
    "val":   [("VOC2007", "test")],                                # 07 test as val
}

VOC_NAMES = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair",
    "cow","diningtable","dog","horse","motorbike","person","pottedplant",
    "sheep","sofa","train","tvmonitor"
]
NAME2ID = {n:i for i,n in enumerate(VOC_NAMES)}

def read_image_set(voc_year, split):
    set_file = Path(VOC_ROOT)/voc_year/"ImageSets"/"Main"/f"{split}.txt"
    with open(set_file, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    objs = []
    for obj in root.findall("object"):
        # optional: skip difficult instances
        diff = obj.findtext("difficult")
        if diff is not None and diff.strip() == "1":
            continue
        name = obj.find("name").text
        if name not in NAME2ID:
            continue
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        # clamp and sanity
        xmin = max(0.0, min(xmin, w-1))
        ymin = max(0.0, min(ymin, h-1))
        xmax = max(0.0, min(xmax, w-1))
        ymax = max(0.0, min(ymax, h-1))
        bw = max(0.0, xmax - xmin)
        bh = max(0.0, ymax - ymin)
        if bw <= 0 or bh <= 0:
            continue
        # YOLO normalized
        xc = (xmin + xmax) / 2.0 / w
        yc = (ymin + ymax) / 2.0 / h
        ww = bw / w
        hh = bh / h
        # clamp to [0,1]
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        ww = min(max(ww, 0.0), 1.0)
        hh = min(max(hh, 0.0), 1.0)
        objs.append((NAME2ID[name], xc, yc, ww, hh))
    return objs

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def main():
    for split_name, sources in SPLITS.items():
        img_out_dir = Path(OUT_ROOT)/"images"/split_name
        lbl_out_dir = Path(OUT_ROOT)/"labels"/split_name
        ensure_dir(img_out_dir); ensure_dir(lbl_out_dir)

        count = 0
        for voc_year, imgset in sources:
            ids = read_image_set(voc_year, imgset)
            img_dir = Path(VOC_ROOT)/voc_year/"JPEGImages"
            ann_dir = Path(VOC_ROOT)/voc_year/"Annotations"
            for img_id in ids:
                src_img = img_dir/f"{img_id}.jpg"
                src_xml = ann_dir/f"{img_id}.xml"
                if not (src_img.exists() and src_xml.exists()):
                    continue
                # parse objects
                objs = parse_xml(src_xml)
                # copy image
                dst_img = img_out_dir/f"{img_id}.jpg"
                shutil.copy2(src_img, dst_img)
                # write label
                dst_lbl = lbl_out_dir/f"{img_id}.txt"
                with open(dst_lbl, "w") as f:
                    for cid, xc, yc, ww, hh in objs:
                        f.write(f"{cid} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")
                count += 1
        print(f"[{split_name}] wrote {count} images (and labels)")

if __name__ == "__main__":
    main()

