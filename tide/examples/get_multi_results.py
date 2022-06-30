import os
import mmcv
from tide.tidecv import TIDE, datasets

gt = '/home/dong/BigDongDATA/DATA/UnderwaterDetection/2020-old/data/annotation_json/val.json'
root_path = '/home/dong/code_sensetime/uwdet/UWdetection/work_dirs/result'
sub_dirs = os.listdir(root_path)
sub_dirs.sort()
print()
for name in sub_dirs:
    if 'bbox' in name:
        out_name = f'{name.split(".")[0]}_tide_error.json'
        result = os.path.join(root_path, name)
        out_name = os.path.join(root_path, out_name)
        tide = TIDE()
        tide.evaluate(datasets.COCO(path=gt), datasets.COCOResult(path=result), mode=TIDE.BOX)
        tide.summarize()
        errors = tide.get_all_errors()

        mmcv.dump(errors, out_name)
        out_dir = os.path.join(root_path, 'pic')
        tide.plot(out_dir=out_dir)
