#!/bin/sh
echo 'disk+superglue_disk'
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./dataset/megadepth/different_scale_overlap/validation_1000/megadepth_scale_2.txt --output_dir outputs/megadepth_2 --matcher superglue_disk --extractor disk-desc  --resize 640 --save --overlaper oetr

echo 'superpoint+superglue'
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./dataset/megadepth/different_scale_overlap/validation_1000/megadepth_scale_2.txt --output_dir outputs/megadepth_2 --matcher superglue_outdoor --extractor superpoint_aachen  --resize 640 --save --overlaper oetr
echo 'superpoint+NN'
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./dataset/megadepth/different_scale_overlap/validation_1000/megadepth_scale_2.txt --output_dir outputs/megadepth_2 --matcher NN --extractor superpoint_aachen  --resize 640 --save --overlaper oetr
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./dataset/megadepth/different_scale_overlap/validation_1000/megadepth_scale_2.txt --output_dir outputs/megadepth_2 --matcher NN --extractor d2net-ss  --resize 640 --save --overlaper oetr
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./dataset/megadepth/different_scale_overlap/validation_1000/megadepth_scale_2.txt --output_dir outputs/megadepth_2 --matcher NN --extractor r2d2-desc  --resize 640 --save --overlaper oetr
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./dataset/megadepth/different_scale_overlap/validation_1000/megadepth_scale_2.txt --output_dir outputs/megadepth_2 --matcher NN --extractor context-desc  --resize 640 --save --overlaper oetr
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./dataset/megadepth/different_scale_overlap/validation_1000/megadepth_scale_2.txt --output_dir outputs/megadepth_2 --matcher NN --extractor aslfeat-desc  --resize 640 --save --overlaper oetr

echo 'loftr'
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./dataset/megadepth/different_scale_overlap/validation_1000/megadepth_scale_2.txt --output_dir outputs/megadepth_2 --matcher loftr --direct --resize 640 --save --overlaper oetr

echo 'cotr landmark'
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./dataset/megadepth/different_scale_overlap/validation_1000/megadepth_scale_2.txt --output_dir outputs/megadepth_2 --matcher cotr --extractor landmark --direct --resize 640 --save --overlaper oetr
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./dataset/megadepth/different_scale_overlap/validation_1000/megadepth_scale_2.txt --output_dir outputs/megadepth_2 --matcher cotr --extractor landmark --resize 640 --save --overlaper oetr
