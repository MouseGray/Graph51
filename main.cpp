#include <iostream>
#include <array>
#include <vector>
#include <set>
#include <algorithm>
#include <fstream>
#include <thread>
#include <numeric>
#include <unistd.h>
#include <immintrin.h>

struct mm_free
{
    void operator()(void* ptr) const noexcept
    {
        _mm_free(ptr);
    }
};

struct Edge { std::size_t A, B; int Length; };

class BitMap
{
    static constexpr unsigned           shift = 5;
    static constexpr unsigned long long lmask = 0x1F;
public:
    BitMap(std::size_t size) :
        size_{size},
        byte_size_{ToBytes(size)},
        data_{std::make_unique<std::uint64_t[]>(ToBytes(size))} {}

    inline void Set(std::size_t idx) noexcept
    {
        data_[idx >> shift] |= 3ull << ((idx & lmask) << 1);
    }

    inline bool Test(std::size_t idx) const noexcept
    {
        return data_[idx >> shift] & (3ull << ((idx & lmask) << 1));
    }

    inline std::size_t Size() const noexcept
    {
        return size_;
    }

    inline uint16_t Mask(std::size_t idx) const noexcept
    {
        return data_[idx >> shift] >> ((idx & lmask) << 1);
    }
private:
    inline std::size_t ToBytes(std::size_t val) const noexcept
    {
        return (val >> shift) + !!(val & lmask);
    }

    std::size_t                      size_, byte_size_;
    std::unique_ptr<std::uint64_t[]> data_;
};

Edge NextEdge(const uint16_t* points, std::size_t size, const BitMap& bitmap)
{
    uint16_t args[16] __attribute__((aligned(32)));
    uint16_t itrs[16] __attribute__((aligned(32)));

    const __m256i arg_set = _mm256_set_epi16(7, 15, 6, 14, 5, 13, 4, 12, 3, 11, 2, 10, 1, 9, 0, 8);

    __m256i min    = _mm256_set1_epi16(std::numeric_limits<short>::max());
    __m256i minarg = _mm256_set1_epi16(0);
    __m256i minitr = _mm256_set1_epi16(0);

    for(std::size_t i = 0; i < bitmap.Size(); ++i)
    {
        if(!bitmap.Test(i))
            continue;

        const auto point_i = i << 1;

        __m256i point = _mm256_set1_epi32((points[point_i | 1] << 16) | points[point_i]);

        for(std::size_t j = 0; j < bitmap.Size(); j += 16)
        {
            __m256i arg = _mm256_add_epi16(arg_set, _mm256_set1_epi16(j));

            uint16_t lmask = bitmap.Mask(j);
            uint16_t hmask = bitmap.Mask(j + 8);

            __m256i mask = _mm256_cmpgt_epi16(_mm256_set_epi16(!!(lmask & 0xC000), !!(hmask & 0xC000), lmask & 0x3000, hmask & 0x3000,
                                                               lmask & 0x0C00, hmask & 0x0C00, lmask & 0x0300, hmask & 0x0300,
                                                               lmask & 0x00C0, hmask & 0x00C0, lmask & 0x0030, hmask & 0x0030,
                                                               lmask & 0x000C, hmask & 0x000C, lmask & 0x0003, hmask & 0x0003),
                                              _mm256_set1_epi16(0x00));

            __m256i lpoints = _mm256_load_si256((__m256i_u*)(points + (j << 1)));
            __m256i hpoints = _mm256_load_si256((__m256i_u*)(points + ((j + 8) << 1)));

            __m256i ldiff = _mm256_or_si256(_mm256_subs_epu16(lpoints, point), _mm256_subs_epu16(point, lpoints));
            __m256i hdiff = _mm256_or_si256(_mm256_subs_epu16(hpoints, point), _mm256_subs_epu16(point, hpoints));

            __m256i lsum  = _mm256_add_epi16(ldiff, _mm256_slli_si256(ldiff, 2));
            __m256i hsum  = _mm256_add_epi16(hdiff, _mm256_srli_si256(hdiff, 2));

            __m256i sum = _mm256_blend_epi16(lsum, hsum, 0x55);

            __m256i msum = _mm256_blendv_epi8(sum, _mm256_set1_epi16(0x7FFF), mask);

            __m256i argmask = _mm256_cmpgt_epi16(min, msum);

            minarg = _mm256_blendv_epi8(minarg, arg, argmask);
            minitr = _mm256_blendv_epi8(minitr, _mm256_set1_epi16(i), argmask);

            min = _mm256_min_epu16(min, msum);
        }
    }

    __m128i lmindata, hmindata;

    _mm256_storeu2_m128i(&lmindata, &hmindata, min);

    int lvalue = _mm_extract_epi32(_mm_minpos_epu16(lmindata), 0);
    int hvalue = _mm_extract_epi32(_mm_minpos_epu16(hmindata), 0);

    _mm256_store_si256((__m256i*)args, minarg);
    _mm256_store_si256((__m256i*)itrs, minitr);

    Edge minimal_edge;

    if( (lvalue & 0xFFFF) < (hvalue & 0xFFFF) )
        minimal_edge = { itrs[(lvalue >> 16) + 8], args[(lvalue >> 16) + 8], (lvalue & 0xFFFF) };
    else
        minimal_edge = { itrs[(lvalue >> 16) + 8], args[hvalue >> 16], (hvalue & 0xFFFF) };

    return minimal_edge;
}

using Tree = std::vector<Edge>;

Tree CreateMinimalTree(const uint16_t* points, std::size_t size, int x, int n)
{
    BitMap bitmap{size};
    bitmap.Set(x);

    Tree tree(n - 1);

    for(auto& edge : tree)
    {
        edge = NextEdge(points, size, bitmap);

        bitmap.Set(edge.A);
        bitmap.Set(edge.B);
    }

    return tree;
}

int Length(const Tree& tree)
{
    return std::accumulate(tree.begin(), tree.end(), 0, [](int val, const Edge& edge){
        return val + edge.Length;
    });
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

void FindMinimalTreeImpl(Tree* minimal_tree, uint16_t* points, std::size_t size,
                         std::size_t begin, std::size_t end)
{
    auto minimal_length = std::numeric_limits<int>::max();
    auto minimal_leafs = std::numeric_limits<int>::max();

    for(int i = begin; i < end; ++i)
    {
        auto tree = CreateMinimalTree(points, size, i, size >> 3);

        auto length = Length(tree);
        auto leafs = LeafsCount(tree);

        if(std::tie(length, leafs) < std::tie(minimal_length, minimal_leafs))
        {
            *minimal_tree = tree;
            minimal_leafs = leafs;
            minimal_length = length;
        }
    }
}

Tree FindMinimalTree(uint16_t* points, std::size_t count,
                     std::size_t cores_count)
{
    std::vector<std::thread> threads;

    std::vector<Tree> minimal_trees(cores_count);

    auto step = count/cores_count;

    for(unsigned int i = 0; i < cores_count; ++i)
    {
        auto begin = i * step;
        auto end = i + 1 < cores_count ? (i + 1) * step : count;

        threads.emplace_back(FindMinimalTreeImpl, &minimal_trees[i], points,
                             count, begin, end);
    }

    for(auto& thread : threads)
        thread.join();

    Tree minimal_tree;
    auto minimal_length = std::numeric_limits<int>::max();
    auto minimal_leafs = std::numeric_limits<int>::max();

    for(const auto& tree : minimal_trees)
    {
        auto length = Length(tree);
        auto leafs = LeafsCount(tree);

        if(std::tie(length, leafs) < std::tie(minimal_length, minimal_leafs))
        {
            minimal_tree = tree;
            minimal_leafs = leafs;
            minimal_length = length;
        }
    }

    return minimal_tree;
}

std::unique_ptr<uint16_t, mm_free> ReadFile(const std::string& file,
                                            std::size_t& count)
{
    std::unique_ptr<uint16_t, mm_free> points;
    std::string                        n, eq;
    unsigned short                     x, y;

    std::ifstream in{file};

    if(!in.is_open())
        throw std::invalid_argument{"File " + file + " not found"};

    in >> n >> eq >> count;

    if(in.fail())
        throw std::invalid_argument{"Invalid file structure"};

    points.reset((uint16_t*)_mm_malloc((count << 1)*sizeof(uint16_t), 32));

    auto* points_p = points.get();

    for(int i = 0; i < count; ++i)
    {
        in >> x >> y;

        if(in.fail())
            throw std::invalid_argument{"Invalid file structure"};

        points_p[(i << 1)] = x;
        points_p[(i << 1) | 1] = y;
    }

    return points;
}

void PrintTree(const Tree& tree)
{
    std::cout << "c Width = " << Length(tree) <<
                 ", Leaf = " << LeafsCount(tree) << "\n";
    std::cout << "p edge = " << ((tree.size() + 1) << 3) <<
                 " " << tree.size() << "\n";

    for(const auto& e : tree)
        std::cout << "e " << e.A << " " << e.B << " " << e.Length << "\n";
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
        auto tree = FindMinimalTree(points.get(), count, cores_count);

        PrintTree(tree);
    }
    catch(const std::exception& ex)
    {
        std::cout << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
