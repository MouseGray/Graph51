#pragma once

#include <cstddef>
#include <memory>

#include "utils.h"

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
