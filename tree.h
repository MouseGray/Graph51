#pragma once

#include <numeric>
#include <set>
#include <vector>

struct Edge { std::size_t A, B; int Length; };

using Tree = std::vector<Edge>;

int Length(const Tree& tree)
{
    return std::accumulate(tree.begin(), tree.end(), 0,
                           [](int val, const Edge& edge){
        return val + edge.Length;
    });
}

int MaxEdge(const Tree& tree)
{
    int max = 0;

    for(auto e : tree)
        if(e.Length > max)
            max = e.Length;


    return max;
}

int LeafsCount(const Tree& tree)
{
    int leaf_count = 0;
    std::set<std::size_t> vertices;

    for(const auto& t : tree)
    {
        vertices.insert(t.A);
        vertices.insert(t.B);
    }

    for(std::size_t v : vertices)
    {
        int count = 0;
        for(const auto& t : tree)
        {
            if(t.A == v || t.B == v)
                ++count;
        }

        if(count == 1)
            ++leaf_count;
    }

    return leaf_count;
}
