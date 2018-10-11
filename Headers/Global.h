/*
 * Global.h
 *
 *  Created on: Mar 24, 2017
 *      Author: vgogate
 */

#ifndef GLOBAL_H_
#define GLOBAL_H_

#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <boost/tokenizer.hpp>

using namespace std;

extern long double exponentialIncrease(long double lambda,
		long double temperature);
extern long double linearIncrease(long double lambda, long double temperature);
extern long double distance(long double i, long double j);
//template<typename T>
void printVector(vector<long double>&v);
extern double normalize(double logp1, double logp2);
extern double fRand(double fMin = 0.000001, double fMax = 1.0);
//double fRand(double fMin, double fMax);
struct WLParameters {
	int maxiter;
	bool stochastic;
	bool l2;
	bool quantization;
	int numclusters;
	int qiterations;
	double learningRate;
	double regularizationConstant;
	double convergence_test;
	int minibatchsize;
	bool verbose;
	size_t seed;
	long double t;
	bool linIncrease;
	bool chouetal16;
	double lambda1;
	int y;
	int u;
	WLParameters() {
		maxiter = 10000;
		stochastic = false;
		l2 = false;
		quantization = false;
		numclusters = 10;
		qiterations = 10;
		learningRate = 0.1;
		regularizationConstant = 1.0;
		convergence_test = 1E-8;
		minibatchsize = 10;
		verbose = false;
		seed = std::random_device()();
		t = 1.0;
		linIncrease = false;
		chouetal16=false;
		lambda1=10.0;
		y=1;
		u=1;
	}
	void printParameters();
};

extern void sort_indexes(vector<int> &y, vector<long double>& x);
typedef vector<vector<bool> > Dataset;

extern bool readDataset(string filename, Dataset& dataset);


#endif /* GLOBAL_H_ */
