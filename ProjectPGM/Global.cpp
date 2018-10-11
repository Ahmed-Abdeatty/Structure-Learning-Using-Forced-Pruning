#include "Global.h"
long double min_probability_value=1e-10;
long double max_probability_value=(long double)1.0 - (long double)1e-10;
long double exponentialIncrease(long double lambda, long double temperature) {
	return lambda * temperature;
}
long double linearIncrease(long double lambda, long double temperature) {
	return lambda + temperature;
}
long double distance(long double i, long double j) {
	return fabs(i - j);
}

//template<typename T>
void printVector(vector<long double>&v) {
	for (auto i = v.begin(); i != v.end(); ++i)
		std::cout << *i << ' ';
	cout << endl;
}
double normalize(double logp1, double logp2) {
	long double normconst = 0.0;
	if (logp1 > logp2) {
		if ((logp2 - logp1) < (-100))
			normconst = logp1;
		else
			normconst = logp1 + log(1.0 + exp(logp2 - logp1));
	} else {
		if ((logp1 - logp2) < (-100))
			normconst = logp2;
		else
			normconst = logp2 + log(1.0 + exp(logp1 - logp2));
	}
	long double xtrue= exp(((long double) logp1) - normconst);
	if (xtrue <min_probability_value){
		xtrue=min_probability_value;
		return xtrue;
	}
	if (xtrue > max_probability_value){
		xtrue=max_probability_value;
		return xtrue;
	}
	return xtrue;
}
double fRand(double fMin, double fMax) {
	double f = (double) rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

void sort_indexes(vector<int> &y, vector<long double>& x) {

	y = vector<int>(x.size());
	std::size_t n(0);
	std::generate(std::begin(y), std::end(y), [&] {return n++;});

	std::sort(std::begin(y), std::end(y),
			[&](int i1, int i2) {return x[i1] < x[i2];});

}

bool readDataset(string filename, Dataset& dataset) {
	ifstream in(filename.c_str());
	if (!in.good()) {
		cerr << "Error:File not present\n";
		return false;
	}
	dataset.clear();
	int i = 0;
	while (in.good()) {
		std::string s;
		std::getline(in, s);
		if (s.empty())
			break;
		dataset.push_back(vector<bool>());
		boost::char_separator<char> sep(",");
		boost::tokenizer<boost::char_separator<char> > tok(s, sep);
		for (boost::tokenizer<boost::char_separator<char> >::iterator iter =
				tok.begin(); iter != tok.end(); iter++) {
			if (*iter == "1") {
				dataset[i].push_back(true);
			} else if (*iter == "0") {
				dataset[i].push_back(false);
			}
		}
		i++;
	}
	if (dataset.empty())
		return false;
	in.close();
	return true;
}

void WLParameters::printParameters()
{
	cout<<"L2 ="<<this->l2<<endl;
	cout<<"Max Iterations ="<<this->maxiter<<endl;
	cout<<"Quantization ="<<this->quantization<<endl;
	cout<<"Learning Rate ="<<this->learningRate<<endl;
	cout<<"Regularization Constant ="<<this->regularizationConstant<<endl;
	cout<<"Number of Clusters ="<<this->numclusters<<endl;
	cout<<"Q Iterations ="<<this->qiterations<<endl;
	cout<<"Mini Batch Size="<<this->minibatchsize<<endl;
	cout<<"Temperature Parameter ="<<this->t<<endl;
	cout<<"Seed ="<<this->seed<<endl;
	cout<<"Linear Increase ="<<this->linIncrease<<endl;
	cout<<"Chouetal16 method ="<<this->chouetal16<<endl;
}
