#pragma once

#include <limits>
#include <thread>

#include "bitmap.h"
#include "tree.h"
#include "vpoint.h"

Edge NextEdge(const VPoints* points, const BitMap& bitmap)
{
    uint16_t args[16] __attribute__((aligned(32)));
    uint16_t itrs[16] __attribute__((aligned(32)));

    uint16_t* xs = points->Xs();
    uint16_t* ys = points->Ys();

    const __m256i arg_set = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6,
                                             5, 4, 3, 2, 1, 0);

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

            __m256i bm_mask = _mm256_set_epi16(!!(bm & 0x8000), bm & 0x4000,
                                                  bm & 0x2000 , bm & 0x1000,
                                                  bm & 0x0800 , bm & 0x0400,
                                                  bm & 0x0200 , bm & 0x0100,
                                                  bm & 0x0080 , bm & 0x0040,
                                                  bm & 0x0020 , bm & 0x0010,
                                                  bm & 0x0008 , bm & 0x0004,
                                                  bm & 0x0002 , bm & 0x0001);

            __m256i mask = _mm256_cmpgt_epi16(bm_mask, _mm256_set1_epi16(0x00));

            __m256i xpoints = _mm256_load_si256((__m256i*)(xs + j));
            __m256i ypoints = _mm256_load_si256((__m256i*)(ys + j));

            __m256i xdiff = _mm256_or_si256(_mm256_subs_epu16(xpoints, point_x),
                                            _mm256_subs_epu16(point_x, xpoints));
            __m256i ydiff = _mm256_or_si256(_mm256_subs_epu16(ypoints, point_y),
                                            _mm256_subs_epu16(point_y, ypoints));

            __m256i sum = _mm256_add_epi16(xdiff, ydiff);

            __m256i masked_sum = _mm256_blendv_epi8(sum, _mm256_set1_epi16(
                std::numeric_limits<short>::max()), mask);

//          __m256i amask = _mm256_cmpgt_epi16(min, masked_sum);
//          __m256i bmask = _mm256_cmpeq_epi16(min, masked_sum);

//          __m256i argmask = _mm256_or_si256(amask, bmask);

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

    if( (lvalue & 0xFFFF) <= (hvalue & 0xFFFF) )
        minimal_edge = { itrs[(lvalue >> 16) + 8], args[(lvalue >> 16) + 8],
                         (lvalue & 0xFFFF) };
    else
        minimal_edge = { itrs[hvalue >> 16], args[hvalue >> 16],
                         (hvalue & 0xFFFF) };

    return minimal_edge;
}

Tree CreateMinimalTree(const VPoints* points, int x, int n)
{
    BitMap bitmap{points->Size()};
    bitmap.Set(x);

    int min_length = std::numeric_limits<int>::max();
    int min_leafs = std::numeric_limits<int>::max();
    int max_edge = 0;

    Tree minimal_tree;
    Tree tree(n - 1);

    for(auto& edge : tree)
    {
        edge = NextEdge(points, bitmap);

        bitmap.Set(edge.B);
    }

    return tree;
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
