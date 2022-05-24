#!/bin/sh
echo 'disk+superglue_disk'
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./dataset/ImageMatching/imc_0.1.txt --output_dir outputs/imc_all --matcher superglue_disk --extractor disk-desc --resize -1 --save

echo 'superpoint+superglue'
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./dataset/ImageMatching/imc_0.1.txt --output_dir outputs/imc_all --matcher superglue_outdoor --extractor superpoint_aachen --resize -1 --save
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./dataset/ImageMatching/imc_0.1.txt --output_dir outputs/imc_all --matcher superglue_outdoor --extractor superpoint_aachen --resize 640 --save --overlaper oetr_imc

echo 'superpoint+NN'
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./dataset/ImageMatching/imc_0.1.txt --output_dir outputs/imc_all --matcher NN --extractor superpoint_aachen  --resize -1 --save
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./dataset/ImageMatching/imc_0.1.txt --output_dir outputs/imc_all --matcher NN --extractor d2net-ss  --resize -1 --save
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./dataset/ImageMatching/imc_0.1.txt --output_dir outputs/imc_all --matcher NN --extractor r2d2-desc  --resize -1 --save
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./dataset/ImageMatching/imc_0.1.txt --output_dir outputs/imc_all --matcher NN --extractor context-desc  --resize -1 --save
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./dataset/ImageMatching/imc_0.1.txt --output_dir outputs/imc_all --matcher NN --extractor aslfeat-desc  --resize -1 --save


python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./dataset/ImageMatching/imc_0.1.txt --output_dir outputs/imc_all --matcher NN --extractor superpoint_aachen  --resize 640 --save --overlaper oetr_imc
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./dataset/ImageMatching/imc_0.1.txt --output_dir outputs/imc_all --matcher NN --extractor d2net-ss  --resize 640 --save --overlaper oetr_imc
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./dataset/ImageMatching/imc_0.1.txt --output_dir outputs/imc_all --matcher NN --extractor r2d2-desc  --resize 640 --save --overlaper oetr_imc
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./dataset/ImageMatching/imc_0.1.txt --output_dir outputs/imc_all --matcher NN --extractor context-desc  --resize 640 --save --overlaper oetr_imc
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./dataset/ImageMatching/imc_0.1.txt --output_dir outputs/imc_all --matcher NN --extractor aslfeat-desc  --resize 640 --save --overlaper oetr_imc

echo 'loftr'
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./dataset/ImageMatching/imc_0.1.txt --output_dir outputs/imc_all --matcher loftr --direct --resize -1 --save
