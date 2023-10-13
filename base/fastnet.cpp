#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <cmath>
#include <cstdio>
#include <set>
#include <queue>
#include <iterator>
#include "fastnet.h"

using namespace std;

typedef set<pair<float, Node*>>::iterator neigh_iterator;

float distance(vector<float> params_0, vector<float> params_1) {
    int length = params_0.size();
    float sum = 0;
    for (int i=0; i<length; i++) {
        sum += pow(params_0[i] - params_1[i], 2);
    }

    return sqrt(sum);
}

float dot_product(vector<float> vec1, vector<float> vec2, vector<float> vec3) {
    float product = 0;
    cout << "gag" << endl;
    for (int i = 0; i < vec1.size(); i++) {
        cout << (vec1[i] - vec3[i]) << endl;
        product += (vec1[i] - vec3[i]) * (vec2[i] - vec3[i]);
    }
    
    return product;
}

Graph::Graph(int N, vector<vector<float>> data, vector<string> colors, vector<float> avg_in, vector<float> var_in, float eps_in) {
    for (int i=0; i<N; i++) {
        Node* node = new Node(i, data[i], colors[i]);
        addNode(node);
    }

    n = N;
    avg = avg_in;
    var = var_in;
    eps = eps_in;
    findED();
}

void Graph::removeNode(Node* node)
{
	nodes.erase(node);
	for (set<Node*>::iterator it = nodes.begin(); it != nodes.end(); it++)
	{
		(*it)->removeNeighbourd(node);
	}
}

void Graph::addEdge(Node* begin, Node* end, float distance)
{
	if (nodes.find(begin) == nodes.end() || nodes.find(end) == nodes.end())
		return;
	begin->addNeighbour(end, distance);
	end->addNeighbour(begin, distance);
}

void Graph::addNode(Node* node)
{
	nodes.insert(node);
}

void Graph::removeEdge(Node* begin, Node* end)
{
	if (nodes.find(begin) == nodes.end() || nodes.find(end) == nodes.end())
		return;
	begin->removeNeighbourd(end);
	end->removeNeighbourd(begin);
}

void Graph::findED() {
    vector<Node*> temp_from;
    vector<Node*> temp_to;
    vector<float> temp_dist;
    float maxval = NULL;

    for (node_iterator it = nodes.begin(); it != nodes.end(); it++) {
        for (node_iterator jt = nodes.begin(); jt != nodes.end(); jt++) {
            if ((*it) == (*jt))
                continue;
            Node* so = *it;
            Node* ta = *jt;
            float temp_distance = distance(so->params, ta->params);
            temp_from.push_back(so);
            temp_to.push_back(ta);
            temp_dist.push_back(temp_distance);
            if (maxval == NULL || maxval < temp_distance) {
                maxval = temp_distance;
            }
        }
    }

    for (int i=0; i<temp_dist.size(); i++) {
        float test = temp_dist[i];
        if (test / maxval <= eps) {
            addEdge(temp_from[i], temp_to[i], temp_dist[i]);
        }
    }
}

void Graph::check_visible_neigh(Node* start_node) {
    queue<Node*> start_nodes;
    start_nodes.push(start_node);
    bool flag = false, result = false;
    pair<float, Node*> neighbour, check_this;
    float value;
    Node* current_node;

    cout << "starting check" << endl;

    while (start_nodes.size() > 0) {
        current_node = start_nodes.front();
        current_node->select = true;
        start_nodes.pop();
        if (current_node->neighbours.size() == 0)
            continue;
        for (set<pair<float, Node*>>::reverse_iterator rit=current_node->nbr_begin(); rit != current_node->nbr_end(); rit++) {
            flag = false;
            check_this = (*rit);
            if (check_this.second->select)
                continue;
            for (set<pair<float, Node*>>::iterator itr=current_node->nb_begin(); itr != current_node->nb_end(); itr++) {
                cout << "i'am here: " << endl;
                neighbour = (*itr);
                result = (check_this.second->name == neighbour.second->name);
                cout << "tut " << result << endl;
                if (result) {
                    continue;
                }
                value = dot_product(current_node->params, check_this.second->params, neighbour.second->params);
                cout << "i'am here: " << value << endl;

                if (value < 0)
                    flag = true;
                    break;
            }

            if (flag)
                removeEdge(current_node, check_this.second);
            else
                start_nodes.push(check_this.second);
        }
    }
}


void Node::addNeighbour(Node* node, float distance)
{
    // pair<Node*, float> neighbour;
    // neighbour.first = node;
    // neighbour.second = distance;
    neighbours.insert({distance, node});
}

void Node::removeNeighbourd(Node* neighbour)
{
    for (neigh_iterator it=nb_begin(); it != nb_end(); it++) {
        if ((*it).second == neighbour)
            neighbours.erase((*it));
            break;
    }
}