#include "Global.h"
#include "Potential.h"
#include "KMeans.h"
#include "MN.h"
#include <array>
Kmeans::Kmeans(vector<Potential>& potentials, int numclusters_) {
	numclusters = numclusters_;
	cluster_centers = vector<long double>(numclusters, 0.0);
	numelements = vector<int>(numclusters, 0);
	cluster_assignment = vector<int>(potentials.size() * 4);
	for (int i = 0; i < cluster_assignment.size(); i++) {
		int randnum = rand() % numclusters;
		cluster_assignment[i] = randnum;
		numelements[randnum]++;
	}
}

void Kmeans::updateClusterCenters(vector<Potential>& potentials,
		vector<Potential>& quantizedpotentials) {
	cluster_centers = vector<long double>(numclusters, 0.0);
	for (int i = 0, k = 0; i < potentials.size(); i++) {
		cluster_centers[cluster_assignment[k++]] += potentials[i].data[0][0];
		cluster_centers[cluster_assignment[k++]] += potentials[i].data[0][1];
		cluster_centers[cluster_assignment[k++]] += potentials[i].data[1][0];
		cluster_centers[cluster_assignment[k++]] += potentials[i].data[1][1];
	}
	for (int i = 0; i < cluster_centers.size(); i++)
	{
		//cout<< "clus= "<<cluster_centers[i]<<endl;
		//c_k[i]=cluster_centers[i]/(long double)numelements[i];
		cluster_centers[i] /= (long double) numelements[i];
	}
	for (int i = 0, k = 0; i < potentials.size(); i++) {
		quantizedpotentials[i].data[0][0] =
				cluster_centers[cluster_assignment[k++]];
		quantizedpotentials[i].data[0][1] =
				cluster_centers[cluster_assignment[k++]];
		quantizedpotentials[i].data[1][0] =
				cluster_centers[cluster_assignment[k++]];
		quantizedpotentials[i].data[1][1] =
				cluster_centers[cluster_assignment[k++]];
	}

}


void Kmeans::quantize(vector<Potential>& potentials,
		vector<Potential>& quantizedpotentials) {
	// Steps:
	// Convert to 1 D array
	// Sort the array
	// Select initial cluster centers based on the sorted array
	// Run Kmeans for 10 iterations
	// Return the result
	//cout<<"Running quantization\n";
	int numpoints = potentials.size() * 4;
	vector<int> idx;
	quantizedpotentials = vector<Potential>(potentials.size());
	vector<long double> sorted_array(potentials.size() * 4);
	for (int i = 0, j = 0; i < potentials.size(); i++) {
		sorted_array[j++] = potentials[i].data[0][0];
		sorted_array[j++] = potentials[i].data[0][1];
		sorted_array[j++] = potentials[i].data[1][0];
		sorted_array[j++] = potentials[i].data[1][1];
	}
	sort_indexes(idx, sorted_array);
	sort(sorted_array.begin(), sorted_array.end());
	numelements = vector<int>(numclusters, numpoints / numclusters);
	cluster_centers = vector<long double>(numclusters, 0.0);
	cluster_assignment = vector<int>(numpoints);
	for (int i = 0; i < (numpoints % numclusters); i++)
		++numelements[i];
	for (int i = 0, k = 0; i < numclusters; i++) {
		for (int j = 0; j < numelements[i]; j++) {
			cluster_assignment[idx[k]] = i;
			cluster_centers[i] += sorted_array[k++];
		}
		cluster_centers[i] /= (long double) numelements[i];
	}
	// Run Kmeans for 100 iterations
	for (int iter = 0; iter < 100; iter++) {
		// Reassign clusters
		int num = 0;
		for (int i = 0; i < (numclusters - 1); i++) {
			// Notice that we just have to reassign boundary points which are at numelements[i]-1 and numelements[i]
			// numelements[i-1] is the last (sorted) point in cluster[i-1] and numelements[i] is the first point in cluster[i]
			// The first loop exits when the boundary point numelements[i-1] cannot be reassigned to cluster[i]
			// The second loop exits when the boundary point numelements[i] cannot be reassigned to cluster[i-1]
			while (distance(sorted_array[num + numelements[i] - 1],
					cluster_centers[i])
					> distance(sorted_array[num + numelements[i] - 1],
							cluster_centers[i + 1])) {
				if (numelements[i] == 1)
					break;
				numelements[i]--;
				numelements[i + 1]++;
			}
			while (distance(sorted_array[num + numelements[i]],
					cluster_centers[i])
					< distance(sorted_array[num + numelements[i]],
							cluster_centers[i + 1])) {
				if (numelements[i + 1] == 1)
					break;
				numelements[i]++;
				numelements[i + 1]--;
			}
			num += numelements[i];
		}
		//Update the cluster centers
		cluster_centers = vector<long double>(numclusters, 0.0);
		int k = 0;
		for (int i = 0; i < numclusters; i++) {
			for (int j = 0; j < numelements[i]; j++) {
				cluster_assignment[idx[k]] = i;
				cluster_centers[i] += sorted_array[k++];
			}
			cluster_centers[i] /= (long double) numelements[i];
		}
	}
	for (int i = 0, k = 0; i < potentials.size(); i++) {
		quantizedpotentials[i].data[0][0] =
				cluster_centers[cluster_assignment[k++]];
		quantizedpotentials[i].data[0][1] =
				cluster_centers[cluster_assignment[k++]];
		quantizedpotentials[i].data[1][0] =
				cluster_centers[cluster_assignment[k++]];
		quantizedpotentials[i].data[1][1] =
				cluster_centers[cluster_assignment[k++]];
	}

}
