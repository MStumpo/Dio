import optuna
import subprocess
import re
from plotly.io import show

def run_network(trial):
	result = subprocess.run("./main --dataset datasets/simple.txt --neuronSize {neuronSize} --timeWindow {timeWindow} --reg {reg} --pos-lr {posA} --neg-lr {negA} \
	 	--decay {decay} --path-decay {path_decay}\
		--epochs 1000 \
		--determinism {determinism} --null-window 0".format(
			neuronSize = str(trial.suggest_int("neuronSize", 6,50)),
			timeWindow = str(trial.suggest_int("timeWindow", 2, 50)),
			reg = str(trial.suggest_float("reg", 0.0000000001, 1.0, log=True)),
			posA = str(trial.suggest_float("posA", 0.0000001, 1.0, log=True)),
			negA = str(trial.suggest_float("negA", 0.0000001, 1.0, log=True)),
			decay = str(trial.suggest_float("decay",0.00001,1.0, log=True)),
			path_decay = str(trial.suggest_float("path_decay",0.01,1.0, log=True)),
			determinism = str(trial.suggest_float("determinism",0,1)),
			#firing_val = str(trial.suggest_float("firing_val",-5,5))
			), shell=True, capture_output=True,text=True)


	if result.returncode != 0:
		print("Command failed:", result.stderr)
	else:
		for line in result.stdout.strip().splitlines():
			match = re.search(r"\|testing\|([^|]+)\|", line)

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


