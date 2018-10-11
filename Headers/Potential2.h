#ifndef POTENTIAL_H_
#define POTENTIAL_H_
#include "Global.h"
struct Potential2 {
	int var1, var2;
	double data[2][2];
	bool iszero[2][2];
	Potential2() {
		var1 = var2 = -1;
		data[0][0] = data[0][1] = data[1][0] = data[1][1] = 0.0;
		iszero[0][0] = iszero[0][1] = iszero[1][0] = iszero[1][1] = true;
	}
	pair<int,int> getPair(){
		if(var1<var2){
			return pair<int,int>(var1,var2);
		}
		else return pair<int,int>(var2,var1);
	}
	int getOtherVar(int var) {
		if (var == var1)
			return var2;
		if (var == var2)
			return var1;
		cerr << "Error var not present";
		return -1;
	}
	double getValue(int v1, int v2, bool assign1, bool assign2) {
		if (var1 == v1 && var2 == v2) {
			return data[assign1][assign2];
		}
		if (var2 == v1 && var1 == v2) {
			return data[assign2][assign1];
		}
		return -1;
	}
	long double getMI(){
		long double px[2],py[2];
		px[0]=data[0][0]+data[0][1];
		px[1]=data[1][0]+data[1][1];
		py[0]=data[0][0]+data[1][0];
		py[1]=data[0][1]+data[1][1];
		long double mi=data[0][0]*(log(data[0][0])-log(px[0])-log(py[0]));
		mi+=data[0][1]*(log(data[0][1])-log(px[0])-log(py[1]));
		mi+=data[1][0]*(log(data[1][0])-log(px[1])-log(py[0]));
		mi+=data[1][1]*(log(data[1][1])-log(px[1])-log(py[1]));
		return mi;

	}
	void addValue(int v1, int v2, bool assign1, bool assign2) {
			if (var1 == v1 && var2 == v2) {
				data[assign1][assign2]+=1.0;
			}
			if (var2 == v1 && var1 == v2) {
				data[assign2][assign1]+=1.0;
			}
		}
	void randomInit() {
		data[0][0] = fRand() * ((rand() % 2) > 0 ? (1.0) : (-1.0));
		data[0][1] = fRand() * ((rand() % 2) > 0 ? (1.0) : (-1.0));
		data[1][0] = fRand() * ((rand() % 2) > 0 ? (1.0) : (-1.0));
		data[1][1] = fRand() * ((rand() % 2) > 0 ? (1.0) : (-1.0));
		iszero[0][0] = iszero[0][1] = iszero[1][0] = iszero[1][1] = false;
	}
	void randomInitFeatures() {
			data[0][0] = fRand() * ((rand() % 2) > 0 ? (1.0) : (-1.0));
			data[0][1] = fRand() * ((rand() % 2) > 0 ? (1.0) : (-1.0));
			data[1][0] = fRand() * ((rand() % 2) > 0 ? (1.0) : (-1.0));
			data[1][1] = 0.0;
			iszero[0][0] = iszero[0][1] = iszero[1][0] = false;
			iszero[1][1]=true;
		}
	void initTo1(){
		data[0][0] = 1.0;
				data[0][1] = 1.0;
				data[1][0] = 1.0;
				data[1][1] = 1.0;
				iszero[0][0] = iszero[0][1] = iszero[1][0] = iszero[1][1] = false;
	}
	void updateUsingGradient(Potential& gradient, double learning_rate) {
		if (!iszero[0][0]) {
			data[0][0] += gradient.data[0][0] * learning_rate;
		}
		if (!iszero[0][1]) {
			data[0][1] += gradient.data[0][1] * learning_rate;
		}
		if (!iszero[1][0]) {
			data[1][0] += gradient.data[1][0] * learning_rate;
		}
		if (!iszero[1][1]) {
			data[1][1] += gradient.data[1][1] * learning_rate;
		}
	}
	void updateUsingSubGradient(double subgradient, double learning_rate) {
		if (!iszero[0][0]) {
			if (data[0][0] > 0.0) {
				data[0][0] = std::max(0.0,
						data[0][0] - subgradient * learning_rate);
			} else if (data[0][0] < 0.0) {
				data[0][0] = std::min(0.0,
						data[0][0] + subgradient * learning_rate);
			}
		}
		if (!iszero[0][1]) {
			if (data[0][1] > 0.0) {
				data[0][1] = std::max(0.0,
						data[0][1] - subgradient * learning_rate);
			} else if (data[0][1] < 0.0) {
				data[0][1] = std::min(0.0,
						data[0][1] + subgradient * learning_rate);
			}
		}
		if (!iszero[1][0]) {
			if (data[1][0] > 0.0) {
				data[1][0] = std::max(0.0,
						data[1][0] - subgradient * learning_rate);
			} else if (data[1][0] < 0.0) {
				data[1][0] = std::min(0.0,
						data[1][0] + subgradient * learning_rate);
			}
		}
		if (!iszero[1][1]) {
			if (data[1][1] > 0.0) {
				data[1][1] = std::max(0.0,
						data[1][1] - subgradient * learning_rate);
			} else if (data[1][1] < 0.0) {
				data[1][1] = std::min(0.0,
						data[1][1] + subgradient * learning_rate);
			}
		}
	}
	void updateUsingSubGradient00(double subgradient, double learning_rate) {
		if (!iszero[0][0]) {
			if (data[0][0] > 0.0) {
				data[0][0] = std::max(0.0,
						data[0][0] - subgradient * learning_rate);
			} else if (data[0][0] < 0.0) {
				data[0][0] = std::min(0.0,
						data[0][0] + subgradient * learning_rate);
			}
		}
	}
	void updateUsingSubGradient01(double subgradient, double learning_rate) {
		if (!iszero[0][1]) {
			if (data[0][1] > 0.0) {
				data[0][1] = std::max(0.0,
						data[0][1] - subgradient * learning_rate);
			} else if (data[0][1] < 0.0) {
				data[0][1] = std::min(0.0,
						data[0][1] + subgradient * learning_rate);
			}
		}
	}
	void updateUsingSubGradient10(double subgradient, double learning_rate) {
		if (!iszero[1][0]) {
			if (data[1][0] > 0.0) {
				data[1][0] = std::max(0.0,
						data[1][0] - subgradient * learning_rate);
			} else if (data[1][0] < 0.0) {
				data[1][0] = std::min(0.0,
						data[1][0] + subgradient * learning_rate);
			}
		}
	}
	void updateUsingSubGradient11(double subgradient, double learning_rate) {
		if (!iszero[1][1]) {
			if (data[1][1] > 0.0) {
				data[1][1] = std::max(0.0,
						data[1][1] - subgradient * learning_rate);
			} else if (data[1][1] < 0.0) {
				data[1][1] = std::min(0.0,
						data[1][1] + subgradient * learning_rate);
			}
		}
	}

};
#endif /* POTENTIAL2_H_ */
