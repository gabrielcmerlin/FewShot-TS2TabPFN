# FewShot-TS2TabPFN

## Prerequisites

conda env export > env.yaml

conda create -n thesis python=3.11 -y
conda activate thesis
pip install git+https://github.com/aeon-toolkit/aeon.git

## Results
Resultados univariados (redux): 
https://github.com/time-series-machine-learning/tsml-eval/tree/main/tsml_eval/publications/y2023/tsc_bakeoff/results/best

Resultados multivariados (mv_core):
https://github.com/aeon-toolkit/multiverse/tree/main/results/multiverse

## Which PC I used

c22 e multirocket: giovani / angela / morsoleto / adilson (16 cores)
tsfresh: magali / kenzo (32 cores)

## How to extract features
mkdir logs
mkdir logs/c22
mkdir logs/tsfresh
mkdir logs/multirocket
nohup python3 -u extract_*.py -c config.yaml > logs/*/**.log &

## How to run

tem q aceitar a licença de cada TabPFN
https://platform.priorlabs.ai/accept-license?hf_repo_id=tabpfn_2_5