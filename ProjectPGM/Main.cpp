#include "Global.h"
#include "MN.h"
#include <boost/program_options.hpp>

int main(int argc, char* argv[]) {
	std::string trainfile, validfile, testfile, structfile;
	int algo_id=1;
	WLParameters wlp;
	namespace po = boost::program_options;
	// Declare the supported options.
	po::options_description desc("Allowed options");
	desc.add_options()("help", "produce help message")("training",
			po::value<string>(&trainfile)->required(),
			"(Required) Training Dataset")("algo",
					po::value<int>(&algo_id),
					"1:L1Struct; 2: Greedy")("validation",
			po::value<string>(&validfile)->required(),
			"(Required) Validation Dataset")("test",
			po::value<string>(&testfile)->required(), "(Required) Test Dataset")("maxiterations,m",
			po::value<int>(&wlp.maxiter), "Maximum Iterations: Gradient Ascent")(
			"stochastic,s", "Use Stochastic or Batch Gradient Ascent")("l2,l",
			"Use L2 regularization")("quantization,q", "Use Quantization")(
			"qnumclusters,k", po::value<int>(&wlp.numclusters),
			"Number of clusters in Quantization")("qiterations,i",
			po::value<int>(&wlp.qiterations),
			"Number of iterations after which cluster centers are updated")(
			"learningRate,r", po::value<double>(&wlp.learningRate),
			"Learning Rate (Gradient Ascent)")("L1Constant,p",
			po::value<double>(&wlp.lambda1), "L1 Regularization Constant")(
			"regularizationConstant,c",
			po::value<double>(&wlp.regularizationConstant),
			"Regularization Constant")("Edge exchanged Constant,y",
					po::value<int>(&wlp.y), "m constant")("Edge delete constant,u",
							po::value<int>(&wlp.u), "k constant")("minibatchsize,b",
			po::value<int>(&wlp.minibatchsize),
			"Mini Batch size for stochastic Ascent")("seed",
			po::value<size_t>(&wlp.seed), "Seed for Repeatability")(
			"regIncreaseParameter,t", po::value<long double>(&wlp.t),
			"Parameter for exponential or linear increase in regularization constant")(
			"linearRegIncrease,a",
			"Default is exponential increase, use this to perform linear increase of regularization constant")(
			"chouetal16", "Use Chou etal. 2016 method")("verbose,v",
			"Verbose output");
	try {
		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);
		if (vm.count("help")) {
			std::cout << desc << '\n';
			exit(-1);
		}
		if (vm.count("stochastic")) {
			wlp.stochastic = true;
		}
		if (vm.count("l2")) {
			wlp.l2 = true;
		}
		if (vm.count("quantization")) {
			wlp.quantization = true;
		}
		if (vm.count("verbose")) {
			wlp.verbose = true;
		}
		if (vm.count("linearRegIncrease")) {
			wlp.linIncrease = true;
		}
		if (vm.count("chouetal16")) {
			wlp.chouetal16 = true;
		}
	} catch (po::error &ex) {
		std::cout << desc << '\n';
		std::cerr << ex.what() << '\n';
		exit(-1);
	}

	srand(wlp.seed);
	Dataset train_data, valid_data, test_data;
	readDataset(trainfile, train_data);
	readDataset(validfile, valid_data);
	readDataset(testfile, test_data);
	MN mn;
	mn.wlp = wlp;
	wlp.printParameters();
	//mn.learnStructL1(train_data, valid_data, test_data);
	if (algo_id==1){
		mn.learnStructL1(train_data, valid_data, test_data);
	} else if (algo_id==2){
		mn.greedyProject(train_data, valid_data, test_data);
	}
	else{
		return -1;
	}
	//mn.writeToUAIFile(structfile);
	mn.learnWeightsL2(train_data,valid_data,test_data);
	cout << "Validation PLL = " << mn.getPLL(valid_data) << endl;
	cout << "Test PLL = " << mn.getPLL(test_data) << endl;
}




