#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <queue>
#include <iterator>
#include "net.h"

using namespace std;

float distance(vector<float> params_0, vector<float> params_1) {
    int length = params_0.size();
    float sum = 0;
    for (int i=0; i<length; i++) {
        sum += pow(params_0[i] - params_1[i], 2);
    }

    return sqrt(sum);
}

Graph::Graph(int N, vector<vector<float>> data, vector<float> colors, float eps_in) {
    source_data = data;
    eps = eps_in;
    count_points = N;
    for (int i = 0;i < N;i++) {
        vector<float> fogRow;
        for(int j=0; j<N; j++)
        {
            fogRow.push_back(0);
        }
        matrix.push_back(fogRow);
        select.push(false)
    }
    findED();
}

void Graph::findED() {
    float maxval = NULL;

    for (int i = 0;i < count_points;i++) {
        for (int j = i + 1;j < count_points;j++) {
            matrix[i][j] = distance(source_data[i], source_data[j]);
            matrix[j][i] = matrix[i][j];
            if (maxval == NULL || maxval < matrix[i][j]) {
                maxval = matrix[i][j];
            }
        }
    }

    for (int i = 0;i < count_points;i++) {
        for (int j = i + 1;j < count_points;j++) {
            if (matrix[i][j] / maxval > eps) {
                matrix[i][j] = 0;
                matrix[j][i] = 0;
            }
        }
    }
}

void Graph::check_visible_neigh(int start_index) {
    queue<int> starters;
    starters.push(start_index);
    int c_index = 0;
    while (starters.size() > 0) {
        c_index = starters.front();
        select[c_index] = true;
        starters.pop();

    }
}