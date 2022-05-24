#!/bin/sh
echo 'without OETR'
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./assets/imc/imc_0.1.txt --output_dir outputs/imc_2021 --matcher superglue_disk --extractor disk-desc  --resize -1 --save
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./assets/imc/imc_0.1.txt --output_dir outputs/imc_2021 --matcher superglue_outdoor --extractor superpoint_aachen  --resize -1 --save
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./assets/imc/imc_0.1.txt --output_dir outputs/imc_2021 --matcher NN --extractor superpoint_aachen  --resize -1 --save
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./assets/imc/imc_0.1.txt --output_dir outputs/imc_2021 --matcher NN --extractor d2net-ss  --resize -1 --save
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./assets/imc/imc_0.1.txt --output_dir outputs/imc_2021 --matcher NN --extractor r2d2-desc  --resize -1 --save
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./assets/imc/imc_0.1.txt --output_dir outputs/imc_2021 --matcher NN --extractor context-desc  --resize -1 --save
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./assets/imc/imc_0.1.txt --output_dir outputs/imc_2021 --matcher NN --extractor aslfeat-desc  --resize -1 --save
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./assets/imc/imc_0.1.txt --output_dir outputs/imc_2021 --matcher loftr --direct --resize -1 --save


echo 'with OETR'
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./assets/imc/imc_0.1.txt --output_dir outputs/imc_2021 --matcher superglue_disk --extractor disk-desc  --resize 640 --save --overlaper oetr_imc
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./assets/imc/imc_0.1.txt --output_dir outputs/imc_2021 --matcher superglue_outdoor --extractor superpoint_aachen  --resize 640 --save --overlaper oetr_imc
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./assets/imc/imc_0.1.txt --output_dir outputs/imc_2021 --matcher NN --extractor superpoint_aachen  --resize 640 --save --overlaper oetr_imc
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./assets/imc/imc_0.1.txt --output_dir outputs/imc_2021 --matcher NN --extractor d2net-ss  --resize 640 --save --overlaper oetr_imc
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./assets/imc/imc_0.1.txt --output_dir outputs/imc_2021 --matcher NN --extractor r2d2-desc  --resize 640 --save --overlaper oetr_imc
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./assets/imc/imc_0.1.txt --output_dir outputs/imc_2021 --matcher NN --extractor context-desc  --resize 640 --save --overlaper oetr_imc
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./assets/imc/imc_0.1.txt --output_dir outputs/imc_2021 --matcher NN --extractor aslfeat-desc  --resize 640 --save --overlaper oetr_imc
python3 evaluation.py --input_dir ./dataset/ImageMatching --input_pairs ./assets/imc/imc_0.1.txt --output_dir outputs/imc_2021 --matcher loftr --direct --resize 640 --save --overlaper oetr_imc
