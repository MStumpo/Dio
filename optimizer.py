import optuna
import subprocess
import re
from plotly.io import show

def run_network(trial):
	result = subprocess.run("./main --dataset datasets/5v3_XNOR.txt --dataset-test datasets/5v3_XNOR_test.txt --neuronSize {neuronSize} --timeWindow {timeWindow} --reg {reg} --pos-lr {posA} --neg-lr {negA} \
	 	--decayPre {decayPre} --decayPost {decayPost}\
		--entropy-factor {entrp_factor} --kernel-size {k_size} --kernel-normalization {k_norm} --epochs {epochs} \
		--determinism {determinism} --null-window 0 --col-only 1".format(
			neuronSize = str(trial.suggest_int("neuronSize", 10,200)),
			timeWindow = str(trial.suggest_int("timeWindow", 2, 50)),
			reg = str(trial.suggest_float("reg", 0.0000000001, 1.0, log=True)),
			posA = str(trial.suggest_float("posA", 0.0, 1.0)),
			negA = str(trial.suggest_float("negA", 0.0, 1.0)),
			decayPre = str(trial.suggest_float("decayPre",0.00001,1.0, log=True)),
			decayPost = str(trial.suggest_float("decayPost",0.00001,1.0, log=True)),
			entrp_factor = str(trial.suggest_float("entrp_factor", -5,5)),
			k_size = str(trial.suggest_int("k_size", 0,5)),
			k_norm = 0,
			epochs = 500,
			determinism = str(trial.suggest_float("determinism",0,1)),
			#firing_val = str(trial.suggest_float("firing_val",-5,5))
			), shell=True, capture_output=True,text=True)


	if result.returncode != 0:
		print("Command failed:", result.stderr)
	else:
		for line in result.stdout.strip().splitlines():
			match = re.search(r"score:\s*(-?\d+\.?\d*)", line)
			if(match):
				score = float(match.group(1))
		return score



def __main__():
	study = optuna.create_study(direction="maximize",
		study_name = "ricky",
		storage="sqlite:///ricky.db",
		load_if_exists = True)
	study.optimize(run_network, n_trials = 500)
	print(study.best_params)
	fig = optuna.visualization.plot_optimization_history(study)
	show(fig)


__main__()


