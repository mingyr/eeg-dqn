python train_agent.py --data_dir=/data/yuming/eeg-rl-data/rt-norm/reinforcement/data-intra-s01 --output_dir=output-intra-s01 --gpus=0
python test_agent.py --data_dir=/data/yuming/eeg-rl-data/rt-norm/reinforcement/data-intra-s01 --checkpoint_dir=output-intra-s01 --output_dir=test-output-intra-s01 --gpus=0

python train_agent.py --data_dir=/data/yuming/eeg-rl-data/rt-norm/reinforcement/data-inter --output_dir=output-inter/1 --gpus=1
python test_agent.py --data_dir=/data/yuming/eeg-rl-data/rt-norm/reinforcement/data-inter --checkpoint_dir=output-inter/2 --output_dir=test-output-inter/2 --gpus=1

# supervised
python data_prep.py --indir=/data/yuming/eeg-rl-data/rt-norm/supervised/data-inter/train-128 --outdir=.
python data_prep.py --indir=/data/yuming/eeg-rl-data/rt-norm/supervised/data-inter/val-128 --outdir=.
python data_prep.py --indir=/data/yuming/eeg-rl-data/rt-norm/supervised/data-inter/test-128 --outdir=.

python train_supervised.py --data_dir=data-supervised --output_dir=output-supervised --gpus=0
python test_supervised.py --data_dir=/data/yuming/eeg-rl-data/rt-norm/reinforcement/data-intra-s01 --checkpoint_dir=output-supervised --output_dir=test-output-supervised --gpus=0

python train_supervised.py --data_dir=data-supervised --output_dir=output-supervised --gpus=0
python test_supervised.py --data_dir=/data/yuming/eeg-rl-data/rt-norm/reinforcement/data-inter --checkpoint_dir=output-supervised --output_dir=test-output-supervised --gpus=0


