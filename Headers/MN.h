#include "Global.h"
#include "Potential.h"
struct MN {
	int numvariables;
	int numpotentials;
	vector<Potential> potentials;
	vector<vector<int> > var2potentials;
	WLParameters wlp;
	MN():numvariables(-1),numpotentials(-1),var2potentials(vector<vector<int> >()),wlp(WLParameters()) {
	}
	MN(string filename);
	void learnWeights(Dataset& dataset_,Dataset& valid_dataset, Dataset& test_dataset);
	void learnChouetal16(Dataset& dataset_,Dataset& valid_dataset, Dataset& test_dataset);
	long double getPLL(Dataset& dataset);
	long double getPLLDiff(Potential& pot, vector<vector<long double> >& xtrue, vector<vector<long double> >& xfalse, Dataset& dataset);
	void getPLLStats(vector<vector<long double> >& xtrue, vector<vector<long double> >& xfalse, Dataset& dataset);
	void makeFullNetwork(int num_variables);
	void writeToUAIFile(string filename_);
	void learnStructL1(Dataset& dataset_,Dataset& valid_dataset, Dataset& test_dataset,bool fullnetwork=true);
	void greedyStruct(Dataset& dataset_,Dataset& valid_dataset, Dataset& test_dataset);
	void greedyProject(Dataset& dataset_,Dataset& valid_dataset, Dataset& test_dataset);
	void addVector(Potential temp);
	vector<Potential> setDiff(vector<Potential> oldpotential);
	bool compare(Potential pot1, Potential pot2);
	void learnWeightsL2(Dataset& dataset_,Dataset& valid_dataset, Dataset& test_dataset);
	void chowliu (Dataset& dataset_, Dataset& valid_dataset,Dataset& test_dataset, vector<long double> scores);


};
