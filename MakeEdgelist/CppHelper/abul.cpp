#include<bits/stdc++.h>
#include <fstream>
#include <iostream>
#include <set>

#include "tqdm.hpp"

using namespace std;

void make_adj_list_of_graph(vector<int> *adj_list, string edge_list_file)
{
    

    ifstream MyReadFile(edge_list_file);
    int toNode, fromNode;
    string _;
    
    while(MyReadFile>>toNode>>fromNode>>_)
    {
            adj_list[toNode].push_back(fromNode);
    }

    return;
}

void is_edge_list_compact(string edge_list_file, int &num_nodes)
{
    

    ifstream MyReadFile(edge_list_file);
    int toNode, fromNode;
    string _;
    set<int, greater<int>> unique_nodes;
    vector<int> nodes_list;
    
    while(MyReadFile>>toNode>>fromNode>>_)
    {
            unique_nodes.insert(toNode);
            unique_nodes.insert(fromNode);
            nodes_list.push_back(toNode);
            nodes_list.push_back(fromNode);
    }

    num_nodes = *max_element(nodes_list.begin(), nodes_list.end()) + 1;

    cout << "Number of nodes " << num_nodes << endl;

    cout << "Max element " << *max_element(nodes_list.begin(), nodes_list.end()) << 
           " Min element " << *min_element(nodes_list.begin(), nodes_list.end()) << endl;

    return;
}

void bfs(int start_node,
            vector<int> *adj_list,
            bool *is_visited,
            int *node_to_hop,
            int cutoff,
            vector<int> &just_see_these_nodes)
{
    queue<int> Q;
    Q.push(start_node);
    is_visited[start_node] = true;
    node_to_hop[start_node] = 0;


    just_see_these_nodes.push_back(start_node);

    while( ! Q.empty() )
    {
        int current_node = Q.front();
        Q.pop();


        for(auto adj_node: adj_list[current_node])
        {

            int hop_of_adj_node = node_to_hop[current_node] + 1;
            if( hop_of_adj_node > cutoff || is_visited[adj_node]) continue;
            is_visited[adj_node] = true;
            node_to_hop[adj_node] = hop_of_adj_node;

            Q.push(adj_node);
            just_see_these_nodes.push_back(adj_node);
        }

    }


}

void write_a_line_to_file_node_hopCount_nodes(int node,
                                            string out_file,
                                            int max_hop,
                                            int *node_to_hop,
                                            vector<int> &just_see_these_nodes,
                                            bool printCommaFirst,
                                            bool printStartBracket,
                                            bool printEndBracket)
{
    vector<int> hop_to_hopNodes[max_hop+1];
    for(int a_node_from_just_see: just_see_these_nodes)
    {
        hop_to_hopNodes[node_to_hop[a_node_from_just_see]].push_back(a_node_from_just_see);
    }

    ofstream MyFile(out_file, ios::app);

    

    
    for(int i=0;i<=max_hop;++i)
    {
        bool doesCurrentHopHaveNodes = !hop_to_hopNodes[i].empty();
        if(doesCurrentHopHaveNodes)
        {
            MyFile << node << ":" << i << ":";

            for(int j = 0; j < hop_to_hopNodes[i].size(); ++j)
            {
                bool isFirstNode = j == 0;

                MyFile << (isFirstNode ? "" : ",") << hop_to_hopNodes[i][j];
                
            }
            MyFile << "\n";
        }        
    }

    MyFile.close();
    return;
}


int main()
{
    string edgelist_file = "acm.edgelist";
    int max_hop = 30;
    int num_nodes;
    cout << num_nodes <<endl;

    is_edge_list_compact(edgelist_file, num_nodes);

    vector<int> adj_list[num_nodes];

    make_adj_list_of_graph(adj_list, edgelist_file);

    int firstNodeTracker = 0;

    for(int i: tq::trange(num_nodes))
    {
        bool is_visited[num_nodes];
        memset(is_visited, false, sizeof(is_visited));

        int node_to_hop[num_nodes];
        memset(node_to_hop, -1, sizeof(node_to_hop));

        vector<int> just_see_these_nodes;

        bfs(i, adj_list, is_visited, node_to_hop, max_hop, just_see_these_nodes);

        write_a_line_to_file_node_hopCount_nodes(i, edgelist_file + ".txt", max_hop, node_to_hop, just_see_these_nodes, firstNodeTracker>0, i==0, i==num_nodes-1);

        firstNodeTracker++;

    }

    return 0;
}