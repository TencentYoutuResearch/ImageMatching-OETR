#!/bin/sh
echo 'without OETR'
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./assets/megadepth/megadepth_scale_34.txt --output_dir outputs/megadepth_34 --matcher superglue_disk --extractor disk-desc  --resize -1 --save
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./assets/megadepth/megadepth_scale_34.txt --output_dir outputs/megadepth_34 --matcher superglue_outdoor --extractor superpoint_aachen  --resize -1 --save
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./assets/megadepth/megadepth_scale_34.txt --output_dir outputs/megadepth_34 --matcher NN --extractor superpoint_aachen  --resize -1 --save
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./assets/megadepth/megadepth_scale_34.txt --output_dir outputs/megadepth_34 --matcher NN --extractor d2net-ss  --resize -1 --save
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./assets/megadepth/megadepth_scale_34.txt --output_dir outputs/megadepth_34 --matcher NN --extractor r2d2-desc  --resize -1 --save
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./assets/megadepth/megadepth_scale_34.txt --output_dir outputs/megadepth_34 --matcher NN --extractor context-desc  --resize -1 --save
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./assets/megadepth/megadepth_scale_34.txt --output_dir outputs/megadepth_34 --matcher NN --extractor aslfeat-desc  --resize -1 --save
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./assets/megadepth/megadepth_scale_34.txt --output_dir outputs/megadepth_34 --matcher loftr --direct --resize -1 --save


echo 'with OETR'
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./assets/megadepth/megadepth_scale_34.txt --output_dir outputs/megadepth_34 --matcher superglue_disk --extractor disk-desc  --resize 640 --save --overlaper oetr
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./assets/megadepth/megadepth_scale_34.txt --output_dir outputs/megadepth_34 --matcher superglue_outdoor --extractor superpoint_aachen  --resize 640 --save --overlaper oetr
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./assets/megadepth/megadepth_scale_34.txt --output_dir outputs/megadepth_34 --matcher NN --extractor superpoint_aachen  --resize 640 --save --overlaper oetr
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./assets/megadepth/megadepth_scale_34.txt --output_dir outputs/megadepth_34 --matcher NN --extractor d2net-ss  --resize 640 --save --overlaper oetr
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./assets/megadepth/megadepth_scale_34.txt --output_dir outputs/megadepth_34 --matcher NN --extractor r2d2-desc  --resize 640 --save --overlaper oetr
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./assets/megadepth/megadepth_scale_34.txt --output_dir outputs/megadepth_34 --matcher NN --extractor context-desc  --resize 640 --save --overlaper oetr
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./assets/megadepth/megadepth_scale_34.txt --output_dir outputs/megadepth_34 --matcher NN --extractor aslfeat-desc  --resize 640 --save --overlaper oetr
python3 evaluation.py --input_dir ./dataset/megadepth --input_pairs ./assets/megadepth/megadepth_scale_34.txt --output_dir outputs/megadepth_34 --matcher loftr --direct --resize 640 --save --overlaper oetr
