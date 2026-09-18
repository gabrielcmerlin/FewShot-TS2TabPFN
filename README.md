# FewShot-TS2TabPFN

conda create -n thesis python=3.11 -y
conda activate thesis
pip install git+https://github.com/aeon-toolkit/aeon.git

Resultados univariados (redux): 
https://github.com/time-series-machine-learning/tsml-eval/tree/main/tsml_eval/publications/y2023/tsc_bakeoff/results/best

Resultados multivariados (mv_core):
ver se atualizaram


mkdir logs
mkdir logs/c22
mkdir logs/tsfresh
mkdir logs/multirocket
nohup python3 -u extract_*.py -c config.yaml > logs/*/**.log &


conda env export > env.yaml