/*
 * KMeans.h
 *
 *  Created on: Mar 24, 2017
 *      Author: vgogate
 */

#ifndef KMEANS_H_
#define KMEANS_H_
#include "Global.h"
struct Kmeans {
	// number of clusters
	int numclusters;
	// cluster centers; vector of size numclusters
	vector<long double> cluster_centers;
	// number of elements in each cluster
	vector<int> numelements;
	// cluster assignments from
	vector<int> cluster_assignment;

	Kmeans(vector<Potential>& potentials, int numclusters_);
	void updateClusterCenters(vector<Potential>& potentials,
			vector<Potential>& quantizedpotentials);
	void quantize(vector<Potential>& potentials,
			vector<Potential>& quantizedpotentials);
};
#endif /* KMEANS_H_ */
