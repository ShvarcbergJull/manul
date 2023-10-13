#pragma once
#include <set>
#include <iterator>
#include <queue>

// typedef std::set<Node*>::iterator node_iterator;

class Node
{
	int name;
    std::vector<float> params;
    std::string color;
	std::set<std::pair<float, Node*>> neighbours;
	bool select = false;
	void addNeighbour(Node* neighbourd, float distance);
	void removeNeighbourd(Node* neighbourd);
public:
	Node(const int& aname) : name(aname)
	{
		name = aname;
	}

    Node(const int& aname, std::vector<float> params_in, std::string color_in) : Node(aname)
	{
		params = params_in;
        color = color_in;
	}

	const int& getName() const
	{
		return name;
	}
	std::set<std::pair<float, Node*>>::iterator nb_begin() const
	{
		return neighbours.begin();
	}
	std::set<std::pair<float, Node*>>::iterator nb_end()  const
	{
		return neighbours.end();
	}

	std::set<std::pair<float, Node*>>::reverse_iterator nbr_begin() const
	{
		return neighbours.rbegin();
	}

	std::set<std::pair<float, Node*>>::reverse_iterator nbr_end()  const
	{
		return neighbours.rend();
	}
    
    std::vector<float> get_params() {
        return this->params;
    }

    std::string get_color() {
        return this->color;
    }
	
	std::set<std::pair<float, Node*>> get_neighs() {
		return neighbours;
	}

	friend class Graph;
};

// struct comp {
// 	bool operator()(std::pair<Node*, float> a, std::pair<Node*, float> b) {
// 		return (a.second > b.second);
// 	}
// };

typedef std::set<Node*>::iterator node_iterator;

class Graph
{
	std::set<Node*> nodes;
    int n;
    std::vector<float> avg, var;
	float eps;
public:
	Graph(int N, std::vector<std::vector<float>> data, std::vector<std::string> colors, std::vector<float> avg_in, std::vector<float> var_in, float eps_in);
	void addNode(Node* node);
	void removeNode(Node* node);
	void addEdge(Node* begin, Node* end, float distance);
	void removeEdge(Node* begin, Node* end);
    void findED();
	void check_visible_neigh(Node* start_node);
    std::set<Node*> get_nodes() {
        return this->nodes;
    }
	node_iterator begin() const
	{
		return nodes.begin();
	}
	node_iterator end() const
	{
		return nodes.end();
	}
};