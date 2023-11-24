#pragma once
#include <vector>


class Graph
{
    std::vector<std::vector<float>> matrix;
    std::vector<std::vector<float>> source_data;
    float eps;
    int count_points;
    std::vector<bool> select;
public:
    Graph(int N, std::vector<std::vector<float>> data, std::vector<float> colors, float eps_in);
    void findED();
    void check_visible_neigh(int start_index);
};