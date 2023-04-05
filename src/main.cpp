#include <algorithm>
#include <fstream>
#include <iostream>

#include "algorithm.h"

VPoints ReadFile(const std::string& file, std::size_t& count)
{
    std::string    n, eq;
    unsigned short x, y;

    std::ifstream in{file};

    if(!in.is_open())
        throw std::invalid_argument{"File " + file + " not found"};

    in >> n >> eq >> count;

    if(in.fail())
        throw std::invalid_argument{"Invalid file structure"};

    VPoints points(count);

    auto* xs = points.Xs();
    auto* ys = points.Ys();

    for(int i = 0; i < count; ++i)
    {
        in >> x >> y;

        if(in.fail())
            throw std::invalid_argument{"Invalid file structure"};

        xs[i] = x;
        ys[i] = y;
    }

    return points;
}

void PrintTree(const Tree& tree)
{
    std::cout << "c Width = " << Length(tree) <<
                 ", Leaf = " << LeafsCount(tree) << ", Max edge = " << MaxEdge(tree) << "\n";
    std::cout << "p edge " << ((tree.size() + 1) << 3) <<
                 " " << tree.size() << "\n";

    for(const auto& e : tree)
    {
       std::cout << "e " << e.A + 1 << " " << e.B + 1 << " " << e.Length << "\n";
    }
}

int main(int argc, char *argv[])
{
    if(argc < 2)
    {
        std::cout << "STree <file>" << std::endl;
        return 0;
    }

    try
    {
        std::size_t count;

        auto cores_count = std::thread::hardware_concurrency();

        auto points = ReadFile(argv[1], count);

        if(cores_count == 0)
        {
            std::cout << "Hardware concurrency not supported" << std::endl;
            return 1;
        }

        auto step = count/cores_count;

        std::cout << "System information:\n";
        std::cout << "   Hardware concurrency=" << cores_count << "\n";
        std::cout << "   Step=" << step << "\n";
        std::cout << std::endl;

        std::cout << "Calculation..." << std::endl;
        auto tree = FindMinimalTree(points, cores_count);

        PrintTree(tree);
    }
    catch(const std::exception& ex)
    {
        std::cout << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
