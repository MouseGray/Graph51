#pragma once

#include <immintrin.h>

struct mm_free
{
    void operator()(void* ptr) const noexcept
    {
        _mm_free(ptr);
    }
};
