import optuna
import subprocess
import re
from plotly.io import show

def run_network(trial):
	result = subprocess.run("./main --neuronSize {neuronSize} --timeWindow {timeWindow} --lr {lr} --reg {reg} --tau-pos {tpos} --tau-neg {tneg} --decay {decay} \
		--entropy-factor {entrp_factor} --kernel-size {k_size} --kernel-normalization {k_norm} --epochs {train_epochs} --test-epochs {test_epochs} \
		--determinism {determinism} --firing-value {firing_val}".format(
			neuronSize = str(trial.suggest_int("neuronSize", 10,100)),
			timeWindow = str(trial.suggest_int("timeWindow", 2, 10)),
			lr = str(trial.suggest_float("lr", 0.000000001,1.0, log=True)),
			reg = str(trial.suggest_float("reg", 0.000000001, 1.0, log=True)),
			tpos = str(trial.suggest_float("tpos", 0.0, 500)),
			tneg = str(trial.suggest_float("tneg", 0.0, 500)),
			decay = str(trial.suggest_float("decay",0.0,1.0)),
			entrp_factor = str(trial.suggest_float("entrp_factor", -5,5)),
			#entrp_factor = 1,
			k_size = str(trial.suggest_int("k_size", 0,5)),
			#k_norm = str(trial.suggest_categorical("k_norm", [0,1])),
			k_norm = 0,
			#train_epochs = str(trial.suggest_int("train_epochs", 1,200)),
			train_epochs = 50,
			#test_epochs = str(trial.suggest_int("test_epochs",1,50))
			test_epochs = 50,
			determinism = str(trial.suggest_float("determinism",0,1)),
			#firing_val = str(trial.suggest_float("firing_val",-5,5))
			firing_val = 1
			), shell=True, capture_output=True,text=True)

	scores = []

	if result.returncode != 0:
		print("Command failed:", result.stderr)
	else:
		for line in result.stdout.strip().splitlines():
			match = re.search(r"score:\s*(-?\d+\.?\d*)", line)
			if(match):
				scores.append(float(match.group(1)))
		return sum(scores) / len(scores)



def __main__():
	study = optuna.create_study(direction="maximize",
		study_name = "Ricky_params_categorical",
		storage="sqlite:///Ricky_params_categorical.db",
		load_if_exists = True)
	study.optimize(run_network, n_trials = 500)
	print(study.best_params)
	fig = optuna.visualization.plot_optimization_history(study)
	show(fig)


__main__()


