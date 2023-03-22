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
#include <bitset>

struct mm_free
{
    void operator()(void* ptr) const noexcept
    {
        _mm_free(ptr);
    }
};

struct Edge { std::size_t A, B; int Length; };

void print_m256i(__m256i val)
{
    uint16_t arr[16] __attribute__((aligned(32)));

    _mm256_storeu_si256((__m256i*)arr, val);

    printf("%d %d %d %d   %d %d %d %d   %d %d %d %d   %d %d %d %d\n",
           arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7],
           arr[8], arr[9], arr[10], arr[11], arr[12], arr[13], arr[14], arr[15]);
}

class VPoints
{
public:
    VPoints(std::size_t size) :
        size_{size}
    {
        auto xs = (uint16_t*)_mm_malloc(size*sizeof(uint16_t), 32);
        if(!xs)
            throw std::bad_alloc{};

        xs_.reset(xs);

        auto ys = (uint16_t*)_mm_malloc(size*sizeof(uint16_t), 32);
        if(!ys)
            throw std::bad_alloc{};

        ys_.reset(ys);
    }

    inline uint16_t* Xs() const noexcept
    {
        return xs_.get();
    }

    inline uint16_t* Ys() const noexcept
    {
        return ys_.get();
    }

    inline std::size_t Size() const noexcept
    {
        return size_;
    }
private:
    std::size_t size_;
    std::unique_ptr<uint16_t, mm_free> xs_;
    std::unique_ptr<uint16_t, mm_free> ys_;
};

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
        data_[idx >> shift] |= 1ull << ((idx & lmask) << 1);
    }

    inline void Mask(std::size_t idx) noexcept
    {
        data_[idx >> shift] |= 2ull << ((idx & lmask) << 1);
    }

    inline void SetAndMask(std::size_t idx) noexcept
    {
        data_[idx >> shift] |= 3ull << ((idx & lmask) << 1);
    }

    inline void Reset(std::size_t idx) noexcept
    {
        data_[idx >> shift] &= ~(1ull << ((idx & lmask) << 1));
    }

    inline void Unmask(std::size_t idx) noexcept
    {
        data_[idx >> shift] &= ~(2ull << ((idx & lmask) << 1));
    }

    inline bool Test(std::size_t idx) const noexcept
    {
        return data_[idx >> shift] & (1ull << ((idx & lmask) << 1));
    }

    inline std::size_t Size() const noexcept
    {
        return size_;
    }

    inline uint32_t Part(std::size_t idx) const noexcept
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

Edge NextEdge(const VPoints* points, const BitMap& bitmap)
{
    uint16_t args[16] __attribute__((aligned(32)));
    uint16_t itrs[16] __attribute__((aligned(32)));

    uint16_t* xs = points->Xs();
    uint16_t* ys = points->Ys();

    const __m256i arg_set = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

    __m256i min    = _mm256_set1_epi16(std::numeric_limits<short>::max());
    __m256i minarg = _mm256_set1_epi16(0);
    __m256i minitr = _mm256_set1_epi16(0);

    for(std::size_t i = 0; i < bitmap.Size(); ++i)
    {
        if(!bitmap.Test(i))
            continue;

        __m256i point_x = _mm256_set1_epi16(xs[i]);
        __m256i point_y = _mm256_set1_epi16(ys[i]);

        for(std::size_t j = 0; j < bitmap.Size(); j += 16)
        {
            __m256i arg = _mm256_add_epi16(arg_set, _mm256_set1_epi16(j));

            uint32_t bm = bitmap.Part(j);

            __m256i bm_mask = _mm256_set_epi16(!!(bm & 0xC0000000), !!(bm & 0x30000000),
                                               !!(bm & 0x0C000000), !!(bm & 0x03000000),
                                               !!(bm & 0x00C00000), !!(bm & 0x00300000),
                                               !!(bm & 0x000C0000), !!(bm & 0x00030000),
                                               !!(bm & 0x0000C000), !!(bm & 0x00003000),
                                               !!(bm & 0x00000C00), !!(bm & 0x00000300),
                                               !!(bm & 0x000000C0), !!(bm & 0x00000030),
                                               !!(bm & 0x0000000C), !!(bm & 0x00000003));

            __m256i mask = _mm256_cmpgt_epi16(bm_mask, _mm256_set1_epi16(0x00));

            __m256i xpoints = _mm256_load_si256((__m256i_u*)(xs + j));
            __m256i ypoints = _mm256_load_si256((__m256i_u*)(ys + j));

            __m256i xdiff = _mm256_or_si256(_mm256_subs_epu16(xpoints, point_x), _mm256_subs_epu16(point_x, xpoints));
            __m256i ydiff = _mm256_or_si256(_mm256_subs_epu16(ypoints, point_y), _mm256_subs_epu16(point_y, ypoints));

            __m256i sum = _mm256_add_epi16(xdiff, ydiff);

            __m256i masked_sum = _mm256_blendv_epi8(sum, _mm256_set1_epi16(std::numeric_limits<short>::max()), mask);

            __m256i argmask = _mm256_cmpgt_epi16(min, masked_sum);

            minarg = _mm256_blendv_epi8(minarg, arg, argmask);
            minitr = _mm256_blendv_epi8(minitr, _mm256_set1_epi16(i), argmask);

            min = _mm256_min_epu16(min, masked_sum);
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

uint16_t Min(const VPoints* points, const BitMap& bitmap)
{
    uint16_t* xs = points->Xs();
    uint16_t* ys = points->Ys();

    __m256i mins = _mm256_set1_epi16(std::numeric_limits<short>::max());

    for(std::size_t i = 0; i < bitmap.Size(); ++i)
    {
        if(!bitmap.Test(i))
            continue;

        __m256i pxs = _mm256_set1_epi16(xs[i]);
        __m256i pys = _mm256_set1_epi16(ys[i]);

        for(std::size_t j = 0; j < bitmap.Size(); j += 16)
        {
            uint32_t bm = bitmap.Part(j);

            __m256i bmmask = _mm256_set_epi16(!!(bm & 0xC0000000), !!(bm & 0x30000000),
                                              !!(bm & 0x0C000000), !!(bm & 0x03000000),
                                              !!(bm & 0x00C00000), !!(bm & 0x00300000),
                                              !!(bm & 0x000C0000), !!(bm & 0x00030000),
                                              !!(bm & 0x0000C000), !!(bm & 0x00003000),
                                              !!(bm & 0x00000C00), !!(bm & 0x00000300),
                                              !!(bm & 0x000000C0), !!(bm & 0x00000030),
                                              !!(bm & 0x0000000C), !!(bm & 0x00000003));

            __m256i mask = _mm256_cmpgt_epi16(bmmask, _mm256_set1_epi16(0x00));

            __m256i psx = _mm256_load_si256((__m256i_u*)(xs + j));
            __m256i psy = _mm256_load_si256((__m256i_u*)(ys + j));

            __m256i dxs = _mm256_or_si256(_mm256_subs_epu16(psx, pxs), _mm256_subs_epu16(pxs, psx));
            __m256i dys = _mm256_or_si256(_mm256_subs_epu16(psy, pys), _mm256_subs_epu16(pys, psy));

            __m256i d = _mm256_add_epi16(dxs, dys);

            __m256i md = _mm256_blendv_epi8(d, _mm256_set1_epi16(std::numeric_limits<short>::max()), mask);

            mins = _mm256_min_epu16(mins, md);
        }
    }

    __m128i lmindata, hmindata;

    _mm256_storeu2_m128i(&lmindata, &hmindata, mins);

    int lvalue = _mm_extract_epi32(_mm_minpos_epu16(lmindata), 0);
    int hvalue = _mm_extract_epi32(_mm_minpos_epu16(hmindata), 0);

    if( (lvalue & 0xFFFF) < (hvalue & 0xFFFF) )
        return lvalue & 0xFFFF;
    else
        return hvalue & 0xFFFF;
}

std::vector<Edge> NextEdges(const VPoints* points, const BitMap& bitmap)
{
    uint16_t args[16] __attribute__((aligned(32)));
    uint16_t itrs[16] __attribute__((aligned(32)));

    uint16_t* xs = points->Xs();
    uint16_t* ys = points->Ys();

    uint16_t min = Min(points, bitmap);

    __m256i mins = _mm256_set1_epi16(min);

    std::vector<Edge> edges;

    for(std::size_t i = 0; i < bitmap.Size(); ++i)
    {
        if(!bitmap.Test(i))
            continue;

        __m256i point_x = _mm256_set1_epi16(xs[i]);
        __m256i point_y = _mm256_set1_epi16(ys[i]);

        for(std::size_t j = 0; j < bitmap.Size(); j += 16)
        {
            uint32_t bm = bitmap.Part(j);

            __m256i bm_mask = _mm256_set_epi16(!!(bm & 0xC0000000), !!(bm & 0x30000000),
                                               !!(bm & 0x0C000000), !!(bm & 0x03000000),
                                               !!(bm & 0x00C00000), !!(bm & 0x00300000),
                                               !!(bm & 0x000C0000), !!(bm & 0x00030000),
                                               !!(bm & 0x0000C000), !!(bm & 0x00003000),
                                               !!(bm & 0x00000C00), !!(bm & 0x00000300),
                                               !!(bm & 0x000000C0), !!(bm & 0x00000030),
                                               !!(bm & 0x0000000C), !!(bm & 0x00000003));

            __m256i mask = _mm256_cmpgt_epi16(bm_mask, _mm256_set1_epi16(0x00));

            __m256i xpoints = _mm256_load_si256((__m256i_u*)(xs + j));
            __m256i ypoints = _mm256_load_si256((__m256i_u*)(ys + j));

            __m256i xdiff = _mm256_or_si256(_mm256_subs_epu16(xpoints, point_x), _mm256_subs_epu16(point_x, xpoints));
            __m256i ydiff = _mm256_or_si256(_mm256_subs_epu16(ypoints, point_y), _mm256_subs_epu16(point_y, ypoints));

            __m256i sum = _mm256_add_epi16(xdiff, ydiff);

            __m256i masked_sum = _mm256_blendv_epi8(sum, _mm256_set1_epi16(std::numeric_limits<short>::max()), mask);

            __m256i arg_mask = _mm256_cmpeq_epi16(mins, masked_sum);

            _mm256_store_si256((__m256i*)args, arg_mask);

            for(int a = 0; a < 16; ++a)
            {
                if(args[a] > 5000)
                    edges.push_back({i, j + a, min});
            }
        }
    }

    return edges;
}

using Tree = std::vector<Edge>;

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


void CreateMinimalTreeImpl(const VPoints* points,
                           Tree& tree, Tree& minimal_tree,
                           int& minimal_length, int& minimal_leafs,
                           BitMap& bitmap, int deep,
                           int max_deep)
{
    if(deep == max_deep)
    {
        if(std::tuple(Length(tree), LeafsCount(tree)) < std::tie(minimal_length, minimal_leafs))
        {
            minimal_tree = tree;
            minimal_length = Length(tree);
            minimal_leafs = LeafsCount(tree);
        }
        return;
    }

    for(auto edge : NextEdges(points, bitmap))
    {
        tree[deep] = edge;

        bitmap.Set(edge.B);

        CreateMinimalTreeImpl(points, tree, minimal_tree, minimal_length,
                              minimal_leafs, bitmap, deep + 1, max_deep);

        bitmap.Reset(edge.B);
    }
}

Tree CreateMinimalTree(const VPoints* points, int x, int n)
{
    BitMap bitmap{points->Size()};
    bitmap.Set(x);

    int min_length = std::numeric_limits<int>::max();
    int min_leafs = std::numeric_limits<int>::max();

    Tree minimal_tree;
    Tree tree(n - 1);

    CreateMinimalTreeImpl(points, tree, minimal_tree, min_length, min_leafs,
                          bitmap, 0, n - 1);

//    for(auto& edge : tree)
//    {
//        edge = NextEdge(points, bitmap);

//        bitmap.Set(edge.B);
//    }

    return minimal_tree;
}

void FindMinimalTreeImpl(Tree* minimal_tree, const VPoints* points,
                         std::size_t begin, std::size_t end)
{
    auto minimal_length = std::numeric_limits<int>::max();
    auto minimal_leafs = std::numeric_limits<int>::max();

    for(int i = begin; i < end; ++i)
    {
        auto tree = CreateMinimalTree(points, i, points->Size() >> 3);

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

Tree FindMinimalTree(const VPoints& points, std::size_t cores_count)
{
    std::vector<std::thread> threads;

    std::vector<Tree> minimal_trees(cores_count);

    auto step = points.Size()/cores_count;

    for(unsigned int i = 0; i < cores_count; ++i)
    {
        auto begin = i * step;
        auto end = i + 1 < cores_count ? (i + 1) * step : points.Size();

        threads.emplace_back(FindMinimalTreeImpl, &minimal_trees[i], &points,
                             begin, end);
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
