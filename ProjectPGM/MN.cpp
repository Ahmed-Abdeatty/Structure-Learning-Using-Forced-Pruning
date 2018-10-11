#include "MN.h"
#include "KMeans.h"
#include <set>
#include <algorithm>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <iostream>
#include <chrono>
#include <map>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
int get_max_index(vector<long double>& vec) {
	long double max = std::numeric_limits<double>::lowest();
	int index = -1;
	for (int i = 0; i < vec.size(); i++) {
		if (vec[i] > max) {
			max = vec[i];
			index = i;
		}
	}
	return index;
}
double ZERO_CONSTANT = 1e-5;
long double getL2Score(long double reg_const, vector<Potential>& potentials,
		vector<Potential>& quantizedpotentials) {
	long double sum = 0.0;
	for (int i = 0; i < potentials.size(); i++) {
		sum +=
				(potentials[i].data[0][0] - quantizedpotentials[i].data[0][0])
						* (potentials[i].data[0][0]
								- quantizedpotentials[i].data[0][0]);
		sum +=
				(potentials[i].data[0][1] - quantizedpotentials[i].data[0][1])
						* (potentials[i].data[0][1]
								- quantizedpotentials[i].data[0][1]);
		sum +=
				(potentials[i].data[1][0] - quantizedpotentials[i].data[1][0])
						* (potentials[i].data[1][0]
								- quantizedpotentials[i].data[1][0]);
		sum +=
				(potentials[i].data[1][1] - quantizedpotentials[i].data[1][1])
						* (potentials[i].data[1][1]
								- quantizedpotentials[i].data[1][1]);
	}
	return sum * reg_const;
}
long double getL1Score(long double reg_const, vector<Potential>& potentials) {
	long double sum = 0.0;
	for (int i = 0; i < potentials.size(); i++) {
		sum += fabs(potentials[i].data[0][0]);
		sum += fabs(potentials[i].data[0][1]);
		sum += fabs(potentials[i].data[1][0]);
		sum += fabs(potentials[i].data[1][1]);
	}
	return sum * reg_const;
}
int count_zeros(vector<Potential>& potentials) {
	int count = 0;
	for (int i = 0; i < potentials.size(); i++) {
		if (fabs(potentials[i].data[0][0]) < ZERO_CONSTANT)
			count++;
		if (fabs(potentials[i].data[0][1]) < ZERO_CONSTANT)
			count++;
		if (fabs(potentials[i].data[1][0]) < ZERO_CONSTANT)
			count++;
		if (fabs(potentials[i].data[1][1]) < ZERO_CONSTANT)
			count++;
	}
	return count;
}
long double gradient_sum(vector<Potential>& potentials) {
	long double sum = 0.0;
	for (int i = 0; i < potentials.size(); i++) {
		sum += fabs(potentials[i].data[0][0]);
		sum += fabs(potentials[i].data[0][1]);
		sum += fabs(potentials[i].data[1][0]);
		sum += fabs(potentials[i].data[1][1]);
	}
	return sum;
}
MN::MN(string filename) :
		MN() {
	ifstream in(filename.c_str());
	if (!in.good()) {
		cerr << "Error:File not present\n";
		return;
	}
	in >> numvariables;
	in >> numpotentials;
	potentials = vector<Potential>(numpotentials);
	for (int i = 0; i < numpotentials; i++) {
		in >> potentials[i].var1;
		in >> potentials[i].var2;
		potentials[i].randomInit();
	}
	var2potentials = vector<vector<int> >(numvariables);
	for (int i = 0; i < numpotentials; i++) {
		var2potentials[potentials[i].var1].push_back(i);
		var2potentials[potentials[i].var2].push_back(i);
	}
	in.close();
}

void MN::writeToUAIFile(string filename_) {
	ofstream out(filename_);
	out << "MARKOV\n";
	out << numvariables << endl;
	for (int i = 0; i < numvariables; i++) {
		out << 2;
		if (i != numvariables - 1)
			out << " ";
	}
	out << endl;
	out << numpotentials << endl;
	for (int i = 0; i < numpotentials; i++) {
		out << 2 << " " << potentials[i].var1 << " " << potentials[i].var2
				<< endl;
	}
	for (int i = 0; i < numpotentials; i++) {
		out << 4 << endl;
		out << exp(potentials[i].data[0][0]) << " "
				<< exp(potentials[i].data[0][1]) << " "
				<< exp(potentials[i].data[1][0]) << " "
				<< exp(potentials[i].data[1][1]) << endl;
	}
	out.close();
}

void MN::makeFullNetwork(int num_variables_) {
	numvariables = num_variables_;
	numpotentials = numvariables * (numvariables - 1) / 2;
	potentials = vector<Potential>(numpotentials);
	int k = 0;
	for (int i = 0; i < numvariables; i++) {
		for (int j = i + 1; j < numvariables; j++) {
			potentials[k].var1 = i;
			potentials[k].var2 = j;
			potentials[k].randomInitFeatures();
			k++;
		}
	}
	var2potentials = vector<vector<int> >(numvariables);
	for (int i = 0; i < numpotentials; i++) {
		var2potentials[potentials[i].var1].push_back(i);
		var2potentials[potentials[i].var2].push_back(i);
	}
}
void MN::learnWeights(Dataset& dataset_, Dataset& valid_dataset,
		Dataset& test_dataset) {
	Dataset dataset = dataset_;
	int nexamples = dataset.size();
	cout << "Dataset size = " << nexamples << endl;
	vector<Potential> quantizedpotentials(numpotentials);
	Kmeans kmeans(potentials, wlp.numclusters);
	if (wlp.stochastic)
		nexamples = wlp.minibatchsize;
	std::minstd_rand rng(wlp.seed);
	std::uniform_int_distribution<int> uni(0, dataset.size() - 1); // guaranteed unbiased
	bool runonce = false;
	double learning_rate = wlp.learningRate;
	double ratio = (double) nexamples / (double) dataset_.size();
	for (int iter = 0; iter < wlp.maxiter; iter++) {
		// Randomly sample examples from the training set
		if (wlp.stochastic) {
			dataset = vector<vector<bool> >(nexamples);
			for (int i = 0; i < nexamples; i++) {
				dataset[i] = dataset_[uni(rng)];
			}
		}
		// Verbose mode is disabled
		vector<vector<long double> > xtrue(nexamples,
				vector<long double>(numvariables, 0.0));
		vector<vector<long double> > xfalse(nexamples,
				vector<long double>(numvariables, 0.0));
		for (int i = 0; i < nexamples; i++) {
			for (int var = 0; var < dataset[i].size(); var++) {
				long double logp1 = 0.0, logp2 = 0.0;
				for (int k = 0; k < var2potentials[var].size(); k++) {
					int pot_id = var2potentials[var][k];
					int otherVar = potentials[pot_id].getOtherVar(var);
					logp1 += potentials[pot_id].getValue(var, otherVar, 1,
							dataset[i][otherVar]);
					logp2 += potentials[pot_id].getValue(var, otherVar, 0,
							dataset[i][otherVar]);
				}
				xtrue[i][var] = normalize(logp1, logp2);
				xfalse[i][var] = 1.0 - xtrue[i][var];
			}
		}
		vector<Potential> gradients(numpotentials);
		for (int i = 0; i < nexamples; i++) {
			for (int feat = 0; feat < numpotentials; feat++) {
				int v1 = potentials[feat].var1;
				int v2 = potentials[feat].var2;
				int assign1 = dataset[i][v1];
				int assign2 = dataset[i][v2];
				gradients[feat].data[assign1][assign2] += 2.0;
				gradients[feat].data[0][assign2] -= xfalse[i][v1];
				gradients[feat].data[1][assign2] -= xtrue[i][v1];
				gradients[feat].data[assign1][0] -= xfalse[i][v2];
				gradients[feat].data[assign1][1] -= xtrue[i][v2];
			}
		}
		for (int i = 0; i < numpotentials; i++) {
			gradients[i].data[0][0] /= (long double) nexamples;
			gradients[i].data[0][1] /= (long double) nexamples;
			gradients[i].data[1][0] /= (long double) nexamples;
			gradients[i].data[1][1] /= (long double) nexamples;
		}

		if (wlp.l2) {
			for (int i = 0; i < numpotentials; i++) {
				gradients[i].data[0][0] -= wlp.regularizationConstant
						* potentials[i].data[0][0] * ratio;
				gradients[i].data[0][1] -= wlp.regularizationConstant
						* potentials[i].data[0][1] * ratio;
				gradients[i].data[1][0] -= wlp.regularizationConstant
						* potentials[i].data[1][0] * ratio;
				gradients[i].data[1][1] -= wlp.regularizationConstant
						* potentials[i].data[1][1] * ratio;
			}
		}
		if (wlp.quantization) {
			// Change clusters after every wlp.qiterations
			if (((iter + 1) % wlp.qiterations) == 0) {
				kmeans.quantize(potentials, quantizedpotentials);
			}
			kmeans.updateClusterCenters(potentials, quantizedpotentials);
			for (int i = 0; i < numpotentials; i++) {
				gradients[i].data[0][0] += wlp.regularizationConstant
						* (quantizedpotentials[i].data[0][0]
								- potentials[i].data[0][0]) * ratio;
				gradients[i].data[0][1] += wlp.regularizationConstant
						* (quantizedpotentials[i].data[0][1]
								- potentials[i].data[0][1]) * ratio;
				gradients[i].data[1][0] += wlp.regularizationConstant
						* (quantizedpotentials[i].data[1][0]
								- potentials[i].data[1][0]) * ratio;
				gradients[i].data[1][1] += wlp.regularizationConstant
						* (quantizedpotentials[i].data[1][1]
								- potentials[i].data[1][1]) * ratio;
			}
		}

		if (((iter + 1) % dataset_.size()) == 0) {
			//cout<<"Learning rate updated from "<<learning_rate<<" to ";
			learning_rate = wlp.learningRate
					/ (1.0 + ((double) (iter + 1) / (double) dataset_.size()));
			//learning_rate=wlp.learningRate;
			cout << "Training set score = "
					<< getPLL(dataset_) * (long double) dataset.size()
							- getL2Score(wlp.regularizationConstant, potentials,
									quantizedpotentials)
							- getL1Score(wlp.lambda1, potentials);
			cout << ", Training set PLL " << getPLL(dataset_)
					<< ", Test set PLL " << getPLL(test_dataset) << endl;
			//cout<<learning_rate<<endl;
			//cout<<"gradient sum = "<<gradient_sum(gradients)<<endl;

		}
		// Update using Gradient
		for (int i = 0; i < numpotentials; i++) {
			potentials[i].updateUsingGradient(gradients[i], learning_rate);
		}

		if (wlp.lambda1 == 0.0) {
			continue;
		}
		// Update using L1 penalty every few iterations (lazy update)
		// lazy update causes a problem because L1 penalty is not applied when the algorithm exits
		// The part of the code where we multiply by mult_constant takes care of this problem
		if (((iter + 1) % dataset_.size()) == 0
				|| ((iter + 1) == wlp.maxiter)) {
			double mult_constant = 1.0;
			if ((iter + 1) == wlp.maxiter)
				mult_constant = (double) ((iter + 1) % dataset_.size())
						/ (double) dataset_.size();
			for (int i = 0; i < numpotentials; i++) {
				potentials[i].updateUsingSubGradient(wlp.lambda1,
						learning_rate * mult_constant);
			}
			if ((iter + 1) == wlp.maxiter)
				cout << "Num zeros = " << count_zeros(potentials)
						<< "; mult-constant = " << mult_constant << endl;
		}
	}
}

void MN::learnChouetal16(Dataset& dataset_, Dataset& valid_dataset,
		Dataset& test_dataset) {
	Dataset dataset = dataset_;
	int nexamples = dataset.size();
	vector<Potential> quantizedpotentials(numpotentials);
	Kmeans kmeans(potentials, wlp.numclusters);
	if (wlp.stochastic)
		nexamples = wlp.minibatchsize;
	//std::random_device rd;     // only used once to initialise (seed) engine
	//std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
	//std::mt19937 rng(wlp.seed);
	std::minstd_rand rng(wlp.seed);
	std::uniform_int_distribution<int> uni(0, dataset.size() - 1); // guaranteed unbiased

	// First run L2
	double learning_rate = wlp.learningRate;
	double ratio = (double) nexamples / (double) dataset_.size();
	for (int iter = 0; iter < (wlp.maxiter / 2); iter++) {
		if (wlp.stochastic) {
			dataset = vector<vector<bool> >(nexamples);
			for (int i = 0; i < nexamples; i++) {
				dataset[i] = dataset_[uni(rng)];
			}
		}
		vector<vector<long double> > xtrue(nexamples,
				vector<long double>(numvariables, 0.0));
		vector<vector<long double> > xfalse(nexamples,
				vector<long double>(numvariables, 0.0));
		for (int i = 0; i < nexamples; i++) {
			for (int var = 0; var < dataset[i].size(); var++) {
				long double logp1 = 0.0, logp2 = 0.0;
				for (int k = 0; k < var2potentials[var].size(); k++) {
					int pot_id = var2potentials[var][k];
					int otherVar = potentials[pot_id].getOtherVar(var);
					logp1 += potentials[pot_id].getValue(var, otherVar, 1,
							dataset[i][otherVar]);
					logp2 += potentials[pot_id].getValue(var, otherVar, 0,
							dataset[i][otherVar]);
				}
				xtrue[i][var] = normalize(logp1, logp2);
				xfalse[i][var] = 1.0 - xtrue[i][var];
			}
		}
		vector<Potential> gradients(numpotentials);
		for (int i = 0; i < nexamples; i++) {
			for (int feat = 0; feat < numpotentials; feat++) {
				int v1 = potentials[feat].var1;
				int v2 = potentials[feat].var2;
				int assign1 = dataset[i][v1];
				int assign2 = dataset[i][v2];
				gradients[feat].data[assign1][assign2] += 2.0;
				gradients[feat].data[0][assign2] -= xfalse[i][v1];
				gradients[feat].data[1][assign2] -= xtrue[i][v1];
				gradients[feat].data[assign1][0] -= xfalse[i][v2];
				gradients[feat].data[assign1][1] -= xtrue[i][v2];
			}
		}
		for (int i = 0; i < numpotentials; i++) {
			gradients[i].data[0][0] /= (long double) nexamples;
			gradients[i].data[0][1] /= (long double) nexamples;
			gradients[i].data[1][0] /= (long double) nexamples;
			gradients[i].data[1][1] /= (long double) nexamples;
		}

		// Run L2
		for (int i = 0; i < numpotentials; i++) {
			gradients[i].data[0][0] -= wlp.regularizationConstant
					* potentials[i].data[0][0] * ratio;
			gradients[i].data[0][1] -= wlp.regularizationConstant
					* potentials[i].data[0][1] * ratio;
			gradients[i].data[1][0] -= wlp.regularizationConstant
					* potentials[i].data[1][0] * ratio;
			gradients[i].data[1][1] -= wlp.regularizationConstant
					* potentials[i].data[1][1] * ratio;
		}
		if (((iter + 1) % dataset_.size()) == 0) {

			learning_rate = wlp.learningRate
					/ (1.0 + ((double) (iter + 1) / (double) dataset_.size()));

		}
		for (int i = 0; i < numpotentials; i++) {
			potentials[i].updateUsingGradient(gradients[i], learning_rate);
		}

		if (wlp.lambda1 == 0.0) {
			continue;
		}
		// Update using L1 penalty every few iterations (lazy update)
		// lazy update causes a problem because L1 penalty is not applied when the algorithm exits
		// The part of the code where we multiply by mult_constant takes care of this problem
		if (((iter + 1) % dataset_.size()) == 0
				|| ((iter + 1) == wlp.maxiter)) {
			double mult_constant = 1.0;
			if ((iter + 1) == wlp.maxiter)
				mult_constant = (double) ((iter + 1) % dataset_.size())
						/ (double) dataset_.size();
			for (int i = 0; i < numpotentials; i++) {
				potentials[i].updateUsingSubGradient(wlp.lambda1,
						learning_rate * mult_constant);
			}
			if ((iter + 1) == wlp.maxiter)
				cout << "Num zeros = " << count_zeros(potentials)
						<< "; mult-constant = " << mult_constant << endl;
		}
	}

	// Quantize the parameters
	kmeans.quantize(potentials, quantizedpotentials);
	// Learn by forcing all quantized parameters to take the same value
	learning_rate = wlp.learningRate;
	for (int iter = 0; iter < (wlp.maxiter / 2); iter++) {
		if (wlp.stochastic) {
			dataset = vector<vector<bool> >(nexamples);
			for (int i = 0; i < nexamples; i++) {
				dataset[i] = dataset_[uni(rng)];
			}
		}
		vector<vector<long double> > xtrue(nexamples,
				vector<long double>(numvariables, 0.0));
		vector<vector<long double> > xfalse(nexamples,
				vector<long double>(numvariables, 0.0));
		for (int i = 0; i < nexamples; i++) {
			for (int var = 0; var < dataset[i].size(); var++) {
				long double logp1 = 0.0, logp2 = 0.0;
				for (int k = 0; k < var2potentials[var].size(); k++) {
					int pot_id = var2potentials[var][k];
					int otherVar = potentials[pot_id].getOtherVar(var);
					logp1 += potentials[pot_id].getValue(var, otherVar, 1,
							dataset[i][otherVar]);
					logp2 += potentials[pot_id].getValue(var, otherVar, 0,
							dataset[i][otherVar]);
				}
				xtrue[i][var] = normalize(logp1, logp2);
				xfalse[i][var] = 1.0 - xtrue[i][var];
			}
		}
		vector<Potential> gradients(numpotentials);
		for (int i = 0; i < nexamples; i++) {
			for (int feat = 0; feat < numpotentials; feat++) {
				int v1 = potentials[feat].var1;
				int v2 = potentials[feat].var2;
				int assign1 = dataset[i][v1];
				int assign2 = dataset[i][v2];
				gradients[feat].data[assign1][assign2] += 2.0;
				gradients[feat].data[0][assign2] -= xfalse[i][v1];
				gradients[feat].data[1][assign2] -= xtrue[i][v1];
				gradients[feat].data[assign1][0] -= xfalse[i][v2];
				gradients[feat].data[assign1][1] -= xtrue[i][v2];
			}
		}
		for (int i = 0; i < numpotentials; i++) {
			gradients[i].data[0][0] /= (long double) nexamples;
			gradients[i].data[0][1] /= (long double) nexamples;
			gradients[i].data[1][0] /= (long double) nexamples;
			gradients[i].data[1][1] /= (long double) nexamples;
		}

		// Hard Clustering
		vector<long double> centers(kmeans.cluster_centers);
		vector<int>& assignment = kmeans.cluster_assignment;
		for (int i = 0, k = 0; i < gradients.size(); i++) {
			centers[assignment[k]] += learning_rate * gradients[i].data[0][0];
			k++;
			centers[assignment[k]] += learning_rate * gradients[i].data[0][1];
			k++;
			centers[assignment[k]] += learning_rate * gradients[i].data[1][0];
			k++;
			centers[assignment[k]] += learning_rate * gradients[i].data[1][1];
			k++;
		}
		if (((iter + 1) % dataset_.size()) == 0) {
			learning_rate = wlp.learningRate
					/ (1.0 + ((double) (iter + 1) / (double) dataset_.size()));
		}
		for (int i = 0, k = 0; i < potentials.size(); i++) {
			potentials[i].data[0][0] = centers[assignment[k++]];
			potentials[i].data[0][1] = centers[assignment[k++]];
			potentials[i].data[1][0] = centers[assignment[k++]];
			potentials[i].data[1][1] = centers[assignment[k++]];
		}
		kmeans.updateClusterCenters(potentials, quantizedpotentials);
	}
}

long double MN::getPLL(Dataset& dataset) {
	long double pll = 0.0;
	for (int i = 0; i < dataset.size(); i++) {
		for (int var = 0; var < dataset[i].size(); var++) {
			long double xtrue = 0.0, xfalse = 0.0;
			for (int k = 0; k < var2potentials[var].size(); k++) {
				int pot_id = var2potentials[var][k];
				int otherVar = potentials[pot_id].getOtherVar(var);
				xtrue += potentials[pot_id].getValue(var, otherVar, 1,
						dataset[i][otherVar]);
				xfalse += potentials[pot_id].getValue(var, otherVar, 0,
						dataset[i][otherVar]);
			}
			xtrue = normalize(xtrue, xfalse);
			xfalse = 1.0 - xtrue;
			if (dataset[i][var])
				pll += log(xtrue);
			else
				pll += log(xfalse);
		}
	}
	return pll / (long double) dataset.size();
}
// PLL Stats are stored in xtrue and xfalse
void MN::getPLLStats(vector<vector<long double> >& xtrue,
		vector<vector<long double> >& xfalse, Dataset& dataset) {
	xtrue = vector<vector<long double> >(dataset.size(),
			vector<long double>(numvariables, 0.0));
	xfalse = vector<vector<long double> >(dataset.size(),
			vector<long double>(numvariables, 0.0));
	for (int i = 0; i < dataset.size(); i++) {
		for (int var = 0; var < dataset[i].size(); var++) {
			long double logp1 = 0.0, logp2 = 0.0;
			for (int k = 0; k < var2potentials[var].size(); k++) {
				int pot_id = var2potentials[var][k];
				int otherVar = potentials[pot_id].getOtherVar(var);
				logp1 += potentials[pot_id].getValue(var, otherVar, 1,
						dataset[i][otherVar]);
				logp2 += potentials[pot_id].getValue(var, otherVar, 0,
						dataset[i][otherVar]);
			}
			xtrue[i][var] = logp1;
			xfalse[i][var] = logp2;
		}
	}
}
long double MN::getPLLDiff(Potential& pot, vector<vector<long double> >& xtrue,
		vector<vector<long double> >& xfalse, Dataset& dataset) {
	long double plldiff = 0.0;

	for (int i = 0; i < dataset.size(); i++) {
		int var;
		long double xtruenew, xfalsenew;
		long double xtrueold, xfalseold;

		var = pot.var1;
		xtrueold = xtrue[i][var];
		xfalseold = xfalse[i][var];
		xtruenew = xtrueold
				+ pot.getValue(var, pot.var2, 1, dataset[i][pot.var2]);
		xfalsenew = xfalseold
				+ pot.getValue(var, pot.var2, 0, dataset[i][pot.var2]);
		xtrueold = normalize(xtrueold, xfalseold);
		xfalseold = 1.0 - xtrueold;
		xtruenew = normalize(xtruenew, xfalsenew);
		xfalsenew = 1.0 - xtruenew;
		if (dataset[i][var])
			plldiff += log(xtruenew) - log(xtrueold);
		else
			plldiff += log(xtruenew) - log(xfalseold);

		var = pot.var2;
		xtrueold = xtrue[i][var];
		xfalseold = xfalse[i][var];
		xtruenew = xtrueold
				+ pot.getValue(var, pot.var1, 1, dataset[i][pot.var1]);
		xfalsenew = xfalseold
				+ pot.getValue(var, pot.var1, 0, dataset[i][pot.var1]);
		xtrueold = normalize(xtrueold, xfalseold);
		xfalseold = 1.0 - xtrueold;
		xtruenew = normalize(xtruenew, xfalsenew);
		xfalsenew = 1.0 - xtruenew;
		if (dataset[i][var])
			plldiff += log(xtruenew) - log(xtrueold);
		else
			plldiff += log(xfalsenew) - log(xfalseold);

	}
	return plldiff;
}
void MN::greedyProject(Dataset& dataset_, Dataset& valid_dataset,Dataset& test_dataset)
{
	std::minstd_rand rng(wlp.seed);
				std::uniform_int_distribution<int> uni(0, dataset_.size() - 1); // guaranteed unbiased
						makeFullNetwork(dataset_[0].size());
						for (int i = 0; i < potentials.size(); i++) {
							potentials[i].initTo1();
						}
						vector<long double> scores(potentials.size());
						for (int i = 0; i < dataset_.size(); i++) {
							for (int var = 0; var < dataset_[i].size(); var++) {
								for (int k = 0; k < var2potentials[var].size(); k++) {
									int pot_id = var2potentials[var][k];
									int otherVar = potentials[pot_id].getOtherVar(var);
									potentials[pot_id].addValue(var, otherVar, dataset_[i][var],
											dataset_[i][otherVar]);
								}
							}
						}
						for (int i = 0; i < potentials.size(); i++) {
							scores[i] = potentials[i].getMI();
						}
						//Initialization for top k potential
						chowliu(dataset_,valid_dataset,test_dataset,scores);
						//computing the set difference
						set<pair<int, int> > remaining_potentials;
						for (int i = 0; i < numvariables; i++)
							for (int j = i + 1; j < numvariables; j++)
								remaining_potentials.insert(pair<int, int>(i, j));
						for (int i = 0; i < potentials.size(); i++) {
							potentials[i].randomInitFeatures();
							remaining_potentials.erase(potentials[i].getPair());
						}
						numpotentials = potentials.size();
						var2potentials = vector<vector<int> >(numvariables);
						for (int i = 0; i < numpotentials; i++) {
							var2potentials[potentials[i].var1].push_back(i);
							var2potentials[potentials[i].var2].push_back(i);
						}
						vector<pair<int, int> > remaining_potentials_vector(
																remaining_potentials.size());
														scores = vector<long double>(remaining_potentials_vector.size());
						long double best_validation_score=std::numeric_limits<long double>::lowest();
						int validation_score_count=0;
						vector<Potential> best_potentials;
						long prev_validation_score=-9999999;
						long double curr_validation_score=getPLL(valid_dataset);
						for (int i = 0; i < wlp.y && !remaining_potentials.empty(); //m potentials adding in the system
																i++)
						{
									int index = get_max_index(scores);
									Potential pot;
									pot.var1 = remaining_potentials_vector[index].first;
									pot.var2 = remaining_potentials_vector[index].second;
									pot.randomInitFeatures();
									potentials.push_back(pot);
									//cout << "Score = " << scores[index] << endl;
									scores[index] = std::numeric_limits<double>::lowest();
									remaining_potentials.erase(pair<int, int>(pot.var1, pot.var2));
						}
						numpotentials = potentials.size();
						var2potentials = vector<vector<int> >(numvariables);
						for (int i = 0; i < numpotentials; i++)
						{
								var2potentials[potentials[i].var1].push_back(i);
								var2potentials[potentials[i].var2].push_back(i);
						}
						while (true) {
								learnWeightsL2(dataset_, valid_dataset, test_dataset);
								//learnChouetal16(dataset_, valid_dataset, test_dataset);
								long double curr_validation_score=getPLL(valid_dataset);
								if (curr_validation_score < best_validation_score){
									validation_score_count=0;
									best_validation_score=curr_validation_score;
									best_potentials=potentials;
								}
								else{
									validation_score_count++;
									if (validation_score_count>5){
										break;
									}
								}
								cout << "Validation PLL = " << curr_validation_score << endl;
								cout << "Test PLL = " << getPLL(test_dataset) << endl;
								Dataset dataset;
								int nexamples = 100;
								dataset = vector<vector<bool> >(nexamples);
								for (int i = 0; i < nexamples; i++) {
									dataset[i] = dataset_[uni(rng)];
								}

								Kmeans kmeans(potentials, wlp.numclusters);
								vector<Potential> quantizedpotentials;
								kmeans.quantize(potentials, quantizedpotentials);
								set<pair<int, int> > remaining_potentials;
								for (int i = 0; i < numvariables; i++)
									for (int j = i + 1; j < numvariables; j++)
										remaining_potentials.insert(pair<int, int>(i, j));
								for (int i = 0; i < potentials.size(); i++) {
									remaining_potentials.erase(potentials[i].getPair());
								}

								// Compute the max score for each remaining potential
								int index = 0;
								vector<vector<long double> > xtrue, xfalse;
								getPLLStats(xtrue, xfalse, dataset);
								for (auto& f : remaining_potentials) {
									remaining_potentials_vector[index] = f;
									Potential pot;
									pot.var1 = f.first;
									pot.var2 = f.second;
									pot.iszero[0][0] = pot.iszero[0][1] = pot.iszero[1][0] =
											pot.iszero[1][1] = false;
									long double score = std::numeric_limits<long double>::lowest();
									for (int i = 0; i < wlp.numclusters; i++) {
										pot.data[0][0] = kmeans.cluster_centers[i];
										score = std::max(score,
												getPLLDiff(pot, xtrue, xfalse, dataset));
									}
									pot.data[0][0] = 0.0;
									for (int i = 0; i < wlp.numclusters; i++) {
										pot.data[0][1] = kmeans.cluster_centers[i];
										score = std::max(score,
												getPLLDiff(pot, xtrue, xfalse, dataset));
									}
									pot.data[0][1] = 0.0;
									for (int i = 0; i < wlp.numclusters; i++) {
										pot.data[1][0] = kmeans.cluster_centers[i];
										score = std::max(score,
												getPLLDiff(pot, xtrue, xfalse, dataset));
									}
									pot.data[1][0] = 0.0;
									for (int i = 0; i < wlp.numclusters; i++) {
										pot.data[1][1] = kmeans.cluster_centers[i];
										score = std::max(score,
												getPLLDiff(pot, xtrue, xfalse, dataset));
									}
									scores[index++] = score;
								}
								cout << "#num potentials = " << potentials.size() << " out of "
										<< numvariables * (numvariables - 1) / 2 << endl;
								if (remaining_potentials.empty())
									return;


							for(int i=0;i<wlp.u;i++) //delete k edges
							{
								potentials.pop_back();
							}
							numpotentials = potentials.size();
							var2potentials = vector<vector<int> >(numvariables);
							for (int i = 0; i < numpotentials; i++)
							{
								var2potentials[potentials[i].var1].push_back(i);
								var2potentials[potentials[i].var2].push_back(i);
							}
							for (int i = 0; i < wlp.u && !remaining_potentials.empty(); //add k edges
																	i++)
							{
								int index = get_max_index(scores);
								Potential pot;
								pot.var1 = remaining_potentials_vector[index].first;
								pot.var2 = remaining_potentials_vector[index].second;
								pot.randomInitFeatures();
								potentials.push_back(pot);
								//cout << "Score = " << scores[index] << endl;
								scores[index] = std::numeric_limits<double>::lowest();
								remaining_potentials.erase(pair<int, int>(pot.var1, pot.var2));
							}
							numpotentials = potentials.size();
							var2potentials = vector<vector<int> >(numvariables);
							for (int i = 0; i < numpotentials; i++)
							{
								var2potentials[potentials[i].var1].push_back(i);
								var2potentials[potentials[i].var2].push_back(i);
							}
						}
							/*potentials = best_potentials;
							numpotentials = potentials.size();
							var2potentials = vector<vector<int> >(numvariables);
							for (int i = 0; i < numpotentials; i++) {
								var2potentials[potentials[i].var1].push_back(i);
								var2potentials[potentials[i].var2].push_back(i);
							}*/
						}


void MN :: chowliu (Dataset& dataset_, Dataset& valid_dataset,Dataset& test_dataset,vector<long double> scores)
{
	 using namespace boost;
	  typedef adjacency_list < vecS, vecS, undirectedS,
	    no_property, property < edge_weight_t, int > > Graph;
	  typedef graph_traits < Graph >::edge_descriptor Edge;
	  typedef std::pair<int, int> E;
	  E edge_array[numpotentials];
	  int weights[numpotentials];
	  for (int i=0; i<potentials.size();i++)
	  {
		  int var1=potentials[i].var1;
		  int var2 =potentials[i].var2;
		  edge_array[i]= E(var1 ,var2);
		  weights[i]=-scores[i];
	  }
	  std::size_t num_edges = sizeof(edge_array) / sizeof(E);
	  Graph g(edge_array, edge_array + num_edges, weights, numvariables);
	  property_map < Graph, edge_weight_t >::type weight = get(edge_weight, g);
	  std::vector < Edge > spanning_tree;
	  kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));
	  potentials.clear();
	  for (std::vector < Edge >::iterator ei = spanning_tree.begin();
	  	         ei != spanning_tree.end(); ++ei)
	  {
	  	     // std::cout << source(*ei, g) << " <--> " << source(*ei, g)
		  Potential temp;
		  temp.var1=source(*ei, g);
		  temp.var2=target(*ei, g);
		  potentials.push_back(temp);

	  }
	 /* const int num_nodes = 5;
	  E edge_array[] = { E(0, 2), E(1, 3), E(1, 4), E(2, 1), E(2, 3),
	    E(3, 4), E(4, 0), E(4, 1)
	  };
	  int weights[] = { 1, 1, 2, 7, 3, 1, 1, 1 };
	    std::size_t num_edges = sizeof(edge_array) / sizeof(E);
	  #if defined(BOOST_MSVC) && BOOST_MSVC <= 1300
	    Graph g(num_nodes);
	    property_map<Graph, edge_weight_t>::type weightmap = get(edge_weight, g);
	    for (std::size_t j = 0; j < num_edges; ++j) {
	      Edge e; bool inserted;
	      boost::tie(e, inserted) = add_edge(edge_array[j].first, edge_array[j].second, g);
	      weightmap[e] = weights[j];
	    }
	  #else
	    Graph g(edge_array, edge_array + num_edges, weights, num_nodes);
	  #endif
	    property_map < Graph, edge_weight_t >::type weight = get(edge_weight, g);
	    std::vector < Edge > spanning_tree;

	    kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));

	    std::cout << "Print the edges in the MST:" << std::endl;
	    for (std::vector < Edge >::iterator ei = spanning_tree.begin();
	         ei != spanning_tree.end(); ++ei) {
	      std::cout << source(*ei, g) << " <--> " << target(*ei, g)
	        << " with weight of " << weight[*ei]
	        << std::endl;
	    }*/

}

void MN::learnWeightsL2(Dataset& dataset_,Dataset& valid_dataset, Dataset& test_dataset) {
	Dataset dataset = dataset_;
	int nexamples = dataset.size();
	vector<Potential> quantizedpotentials(numpotentials);
	Kmeans kmeans(potentials, wlp.numclusters);
	if (wlp.stochastic)
		nexamples = wlp.minibatchsize;
	//std::random_device rd;     // only used once to initialise (seed) engine
	//std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
	//std::mt19937 rng(wlp.seed);
	std::minstd_rand rng(wlp.seed);
	std::uniform_int_distribution<int> uni(0, dataset.size() - 1); // guaranteed unbiased
	bool runonce = false;
	for (int iter = 0; iter < wlp.maxiter; iter++) {
		if (((iter + 1) % wlp.qiterations) == 0){
			cout<<"Updated Regularization Constant ="<<wlp.regularizationConstant<<endl;
			cout<<" Validation PLL at iteration = "<<iter+1<<" is = "<<getPLL(valid_dataset)<<endl;
			cout<<" Test PLL at iteration = "<<iter+1<<" is = "<<getPLL(test_dataset)<<endl;
		}
		//if (iter %10==0)cout<<"Iteration "<<iter<<endl;
		if (wlp.stochastic) {
			dataset = vector<vector<bool> >(nexamples);
			for (int i = 0; i < nexamples; i++) {
				dataset[i] = dataset_[uni(rng)];
			}
		}
		if (wlp.verbose && (iter % 10000 == 0)) {
			//cerr<<"Iteration number "<<iter<<endl;
			long double pll = getPLL(dataset_);
			if (wlp.quantization) {
				long double regularization_term = 0.0;
				for (int i = 0; i < numpotentials; i++) {
					regularization_term += (quantizedpotentials[i].data[0][0]
							- potentials[i].data[0][0])
							* (quantizedpotentials[i].data[0][0]
									- potentials[i].data[0][0])
							/ (double) numpotentials;
					regularization_term += (quantizedpotentials[i].data[0][1]
							- potentials[i].data[0][1])
							* (quantizedpotentials[i].data[0][1]
									- potentials[i].data[0][1])
							/ (double) numpotentials;
					regularization_term += (quantizedpotentials[i].data[1][0]
							- potentials[i].data[1][0])
							* (quantizedpotentials[i].data[1][0]
									- potentials[i].data[1][0])
							/ (double) numpotentials;
					regularization_term += (quantizedpotentials[i].data[1][1]
							- potentials[i].data[1][1])
							* (quantizedpotentials[i].data[1][1]
									- potentials[i].data[1][1])
							/ (double) numpotentials;
				}
				regularization_term *= -wlp.regularizationConstant;
				cout << pll << " " << regularization_term << " "
						<< pll + regularization_term << endl;
			} else {
				if (wlp.l2) {
					long double regularization_term = 0.0;
					for (int i = 0; i < numpotentials; i++) {
						regularization_term -= (potentials[i].data[0][0])
								* (potentials[i].data[0][0])
								/ (double) numpotentials;
						regularization_term -= (potentials[i].data[0][1])
								* (potentials[i].data[0][1])
								/ (double) numpotentials;
						regularization_term -= (potentials[i].data[1][0])
								* (potentials[i].data[1][0])
								/ (double) numpotentials;
						regularization_term -= (potentials[i].data[1][1])
								* (potentials[i].data[1][1])
								/ (double) numpotentials;

					}
					regularization_term *= wlp.regularizationConstant;
					cout << pll << " " << regularization_term << " "
							<< pll + regularization_term << endl;

				} else {
					cout << pll << endl;
				}
			}
		}
		vector<vector<long double> > xtrue(nexamples,
				vector<long double>(numvariables, 0.0));
		vector<vector<long double> > xfalse(nexamples,
				vector<long double>(numvariables, 0.0));
		for (int i = 0; i < nexamples; i++) {
			for (int var = 0; var < dataset[i].size(); var++) {
				long double logp1 = 0.0, logp2 = 0.0;
				for (int k = 0; k < var2potentials[var].size(); k++) {
					int pot_id = var2potentials[var][k];
					int otherVar = potentials[pot_id].getOtherVar(var);
					logp1 += potentials[pot_id].getValue(var, otherVar, 1,
							dataset[i][otherVar]);
					logp2 += potentials[pot_id].getValue(var, otherVar, 0,
							dataset[i][otherVar]);
				}
				xtrue[i][var] = normalize(logp1, logp2);
				xfalse[i][var] = 1.0 - xtrue[i][var];
			}
		}
		vector<Potential> gradients(numpotentials);
		for (int i = 0; i < nexamples; i++) {
			for (int feat = 0; feat < numpotentials; feat++) {
				int v1 = potentials[feat].var1;
				int v2 = potentials[feat].var2;
				int assign1 = dataset[i][v1];
				int assign2 = dataset[i][v2];
				gradients[feat].data[assign1][assign2] += 2.0;
				gradients[feat].data[0][assign2] -= xfalse[i][v1];
				gradients[feat].data[1][assign2] -= xtrue[i][v1];
				gradients[feat].data[assign1][0] -= xfalse[i][v2];
				gradients[feat].data[assign1][1] -= xtrue[i][v2];
			}
		}
		for (int i = 0; i < numpotentials; i++) {
			gradients[i].data[0][0] /= (long double) nexamples;
			gradients[i].data[0][1] /= (long double) nexamples;
			gradients[i].data[1][0] /= (long double) nexamples;
			gradients[i].data[1][1] /= (long double) nexamples;
		}
		if (wlp.l2) {
			for (int i = 0; i < numpotentials; i++) {
				gradients[i].data[0][0] -= wlp.regularizationConstant
						* potentials[i].data[0][0] / (double) numpotentials;
				gradients[i].data[0][1] -= wlp.regularizationConstant
						* potentials[i].data[0][1] / (double) numpotentials;
				gradients[i].data[1][0] -= wlp.regularizationConstant
						* potentials[i].data[1][0] / (double) numpotentials;
				gradients[i].data[1][1] -= wlp.regularizationConstant
						* potentials[i].data[1][1] / (double) numpotentials;
			}
		}
		if (wlp.quantization) {
			if (wlp.regularizationConstant > 10000.0){
				if (((iter + 1) % wlp.qiterations) == 0) {
					if (wlp.verbose){
						cout<<"Hard Clustering\n";
						printVector(kmeans.cluster_centers);
					}
				}
				// Hard Clustering
				vector<long double> centers(kmeans.cluster_centers);
				vector<int> assignment=kmeans.cluster_assignment;
				for (int i = 0, k = 0; i < gradients.size(); i++) {
					centers[assignment[k]] += wlp.learningRate
							*gradients[i].data[0][0]/(long double) kmeans.numelements[assignment[k]];
					k++;
					centers[assignment[k]] += wlp.learningRate
							*gradients[i].data[0][1]/(long double) kmeans.numelements[assignment[k]];
					k++;
					centers[assignment[k]] += wlp.learningRate
							*gradients[i].data[1][0]/(long double) kmeans.numelements[assignment[k]];
					k++;
					centers[assignment[k]] += wlp.learningRate
							*gradients[i].data[1][1]/(long double) kmeans.numelements[assignment[k]];
					k++;
				}
				for (int i = 0, k = 0; i < potentials.size(); i++) {
					potentials[i].data[0][0]=centers[assignment[k++]];
					potentials[i].data[0][1]=centers[assignment[k++]];
					potentials[i].data[1][0]=centers[assignment[k++]];
					potentials[i].data[1][1]=centers[assignment[k++]];
				}
				kmeans.updateClusterCenters(potentials, quantizedpotentials);
				continue;
			}
			// Change clusters after every wlp.qiterations
			if (((iter + 1) % wlp.qiterations) == 0) {
				kmeans.quantize(potentials, quantizedpotentials);
				if (wlp.verbose){
					cerr<<wlp.regularizationConstant<<endl;
					printVector(kmeans.cluster_centers);
				}
				if (wlp.linIncrease) {
					wlp.regularizationConstant = linearIncrease(
							wlp.regularizationConstant, wlp.t);
				} else {
					wlp.regularizationConstant = exponentialIncrease(
							wlp.regularizationConstant, wlp.t);
				}
			}
			kmeans.updateClusterCenters(potentials, quantizedpotentials);
			for (int i = 0; i < numpotentials; i++) {
				gradients[i].data[0][0] += wlp.regularizationConstant
						* (quantizedpotentials[i].data[0][0]
								- potentials[i].data[0][0])
						/ (double) numpotentials;
				gradients[i].data[0][1] += wlp.regularizationConstant
						* (quantizedpotentials[i].data[0][1]
								- potentials[i].data[0][1])
						/ (double) numpotentials;
				gradients[i].data[1][0] += wlp.regularizationConstant
						* (quantizedpotentials[i].data[1][0]
								- potentials[i].data[1][0])
						/ (double) numpotentials;
				gradients[i].data[1][1] += wlp.regularizationConstant
						* (quantizedpotentials[i].data[1][1]
								- potentials[i].data[1][1])
						/ (double) numpotentials;
			}
		}
		//long double gradientsum=0.0;
		for (int i = 0; i < numpotentials; i++) {
			potentials[i].data[0][0] += wlp.learningRate
					* gradients[i].data[0][0];
			potentials[i].data[0][1] += wlp.learningRate
					* gradients[i].data[0][1];
			potentials[i].data[1][0] += wlp.learningRate
					* gradients[i].data[1][0];
			potentials[i].data[1][1] += wlp.learningRate
					* gradients[i].data[1][1];
			//gradientsum+=gradients[i].data[0][0]+gradients[i].data[0][1]+gradients[i].data[1][0]+gradients[i].data[1][1];

		}
	}
}



vector<Potential> MN:: setDiff(vector<Potential> oldpotential)
{
	vector<int> index;
	int count=0;
	for(int i=0;i<oldpotential.size();i++)
	{
		for(int j=0;j<oldpotential.size();j++)
		{
			bool k=compare(potentials[i],oldpotential[j]);
			if(k==true)
			{
				index.push_back(j);
				count++;
			}
		}
	}
	int numpotentiallocal=oldpotential.size();
	for(int i=0;i<index.size();i++)
	{
		int val=index[i];
		int vecEnd=oldpotential.size();
		swap(oldpotential[val],oldpotential[vecEnd-1]);
		oldpotential.pop_back();
		numpotentiallocal=numpotentiallocal-1;
	}
	return oldpotential;
}
bool MN:: compare(Potential pot1, Potential pot2)
{
	//cout<<pot1.var1<<" "<<pot2.var1<<","<<pot1.var2<<" "<<pot2.var2<<endl;
	if((pot1.var1==pot2.var1) && (pot1.var2==pot2.var2))
	{
		return true;
	}
	else
		return false;
}
void MN ::addVector(Potential temp)
{
	potentials.push_back(temp);
	numpotentials=numpotentials+1;
	var2potentials = vector<vector<int> >(numvariables);
    for (int i = 0; i < numpotentials; i++)
    {
    	var2potentials[potentials[i].var1].push_back(i);
		var2potentials[potentials[i].var2].push_back(i);
	}
}
void MN::learnStructL1(Dataset& dataset_, Dataset& valid_dataset,
		Dataset& test_dataset, bool fullnetwork) {

	if (fullnetwork)
		makeFullNetwork(dataset_[0].size());

	learnWeights(dataset_, valid_dataset, test_dataset);
	cout << "Num zeros = " << count_zeros(potentials) << endl;
	vector<Potential> new_potentials;
	int numcounts = 0;
	for (int i = 0; i < potentials.size(); i++) {
		int count = 0;
		if (fabs(potentials[i].data[0][0]) < ZERO_CONSTANT) {
			potentials[i].data[0][0] = 0.0;
			potentials[i].iszero[0][0] = true;
			count++;
		}
		if (fabs(potentials[i].data[0][1]) < ZERO_CONSTANT) {
			potentials[i].data[0][1] = 0.0;
			potentials[i].iszero[0][1] = true;
			count++;
		}
		if (fabs(potentials[i].data[1][0]) < ZERO_CONSTANT) {
			potentials[i].data[1][0] = 0.0;
			potentials[i].iszero[1][0] = true;
			count++;
		}
		if (fabs(potentials[i].data[1][1]) < ZERO_CONSTANT) {
			potentials[i].data[1][1] = 0.0;
			potentials[i].iszero[1][1] = true;
			count++;
		}
		numcounts += count;
		if (count > 3) {
			continue;
		} else {
			new_potentials.push_back(potentials[i]);
		}
	}
	cout << "number of pruned features = " << numcounts - numpotentials
			<< " out of " << numpotentials * 3 << endl;
	potentials = new_potentials;
	numpotentials = potentials.size();
	var2potentials = vector<vector<int> >(numvariables);
	for (int i = 0; i < numpotentials; i++) {
		var2potentials[potentials[i].var1].push_back(i);
		var2potentials[potentials[i].var2].push_back(i);
	}
	cout << "Validation PLL = " << getPLL(valid_dataset) << endl;
	cout << "Test PLL = " << getPLL(test_dataset) << endl;
	//wlp.maxiter *= 10;
	// change the lambda1 penalty to zero so that we don't use it for weight learning
	double old_lambda1 = wlp.lambda1;
	wlp.lambda1 = 0.0;
	cout << "Learning weights again\n";
	//learnChouetal16(dataset_, valid_dataset, test_dataset);
	learnWeights(dataset_, valid_dataset, test_dataset);
	cout << "Validation PLL = " << getPLL(valid_dataset) << endl;
	cout << "Test PLL = " << getPLL(test_dataset) << endl;
	/*
	 learnChouetal16(dataset_, valid_dataset, test_dataset);
	 cout << "Validation PLL = " << getPLL(valid_dataset) << endl;
	 cout << "Test PLL = " << getPLL(test_dataset) << endl;
	 */
	wlp.lambda1 = old_lambda1;
}

/* Extra code
 *
 *
 */
// Update using Block L1 penalty
/*
 vector<double> block_l1_den(kmeans.numclusters, 0.0);
 for (int i = 0, j = 0; i < numpotentials; i++) {
 block_l1_den[kmeans.cluster_assignment[j++]] +=
 potentials[i].data[0][0] * potentials[i].data[0][0];
 block_l1_den[kmeans.cluster_assignment[j++]] +=
 potentials[i].data[0][1] * potentials[i].data[0][1];
 block_l1_den[kmeans.cluster_assignment[j++]] +=
 potentials[i].data[1][0] * potentials[i].data[1][0];
 block_l1_den[kmeans.cluster_assignment[j++]] +=
 potentials[i].data[1][1] * potentials[i].data[1][1];
 }
 for (int i = 0; i < kmeans.numclusters; i++) {
 block_l1_den[i] = sqrt(block_l1_den[i]);
 cout << block_l1_den[i] << " ";
 }
 cout << endl;
 for (int i = 0; i < kmeans.numclusters; i++) {
 cout << kmeans.cluster_centers[i] << " ";
 }
 cout << endl;

 for (int i = 0, j = 0; i < numpotentials; i++) {
 potentials[i].updateUsingSubGradient00(
 (wlp.lambda1 / (double) numpotentials)
 * (potentials[i].data[0][0]
 / block_l1_den[kmeans.cluster_assignment[j++]]),
 wlp.learningRate);
 potentials[i].updateUsingSubGradient01(
 (wlp.lambda1 / (double) numpotentials)
 * (potentials[i].data[0][1]
 / block_l1_den[kmeans.cluster_assignment[j++]]),
 wlp.learningRate);
 potentials[i].updateUsingSubGradient10(
 (wlp.lambda1 / (double) numpotentials)
 * (potentials[i].data[1][0]
 / block_l1_den[kmeans.cluster_assignment[j++]]),
 wlp.learningRate);
 potentials[i].updateUsingSubGradient11(
 (wlp.lambda1 / (double) numpotentials)
 * (potentials[i].data[1][1]
 / block_l1_den[kmeans.cluster_assignment[j++]]),
 wlp.learningRate);

 }

 //std::random_device rd;     // only used once to initialise (seed) engine
 //std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
 //std::mt19937 rng(wlp.seed);
 */
/* Extra code
 *
 *
 */
// Update using Block L1 penalty
/*
 vector<double> block_l1_den(kmeans.numclusters, 0.0);
 for (int i = 0, j = 0; i < numpotentials; i++) {
 block_l1_den[kmeans.cluster_assignment[j++]] +=
 potentials[i].data[0][0] * potentials[i].data[0][0];
 block_l1_den[kmeans.cluster_assignment[j++]] +=
 potentials[i].data[0][1] * potentials[i].data[0][1];
 block_l1_den[kmeans.cluster_assignment[j++]] +=
 potentials[i].data[1][0] * potentials[i].data[1][0];
 block_l1_den[kmeans.cluster_assignment[j++]] +=
 potentials[i].data[1][1] * potentials[i].data[1][1];
 }
 for (int i = 0; i < kmeans.numclusters; i++) {
 block_l1_den[i] = sqrt(block_l1_den[i]);
 cout << block_l1_den[i] << " ";
 }
 cout << endl;
 for (int i = 0; i < kmeans.numclusters; i++) {
 cout << kmeans.cluster_centers[i] << " ";
 }
 cout << endl;

 for (int i = 0, j = 0; i < numpotentials; i++) {
 potentials[i].updateUsingSubGradient00(
 (wlp.lambda1 / (double) numpotentials)
 * (potentials[i].data[0][0]
 / block_l1_den[kmeans.cluster_assignment[j++]]),
 wlp.learningRate);
 potentials[i].updateUsingSubGradient01(
 (wlp.lambda1 / (double) numpotentials)
 * (potentials[i].data[0][1]
 / block_l1_den[kmeans.cluster_assignment[j++]]),
 wlp.learningRate);
 potentials[i].updateUsingSubGradient10(
 (wlp.lambda1 / (double) numpotentials)
 * (potentials[i].data[1][0]
 / block_l1_den[kmeans.cluster_assignment[j++]]),
 wlp.learningRate);
 potentials[i].updateUsingSubGradient11(
 (wlp.lambda1 / (double) numpotentials)
 * (potentials[i].data[1][1]
 / block_l1_den[kmeans.cluster_assignment[j++]]),
 wlp.learningRate);

 }

 //std::random_device rd;     // only used once to initialise (seed) engine
 //std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
 //std::mt19937 rng(wlp.seed);
 */
