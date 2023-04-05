#pragma once

#include <cstdint>
#include <memory>

class BitMap
{
    static constexpr unsigned           shift = 6;
    static constexpr unsigned long long lmask = 0x3F;
public:
    BitMap(std::size_t size) :
        size_{size},
        byte_size_{ToBytes(size)},
        data_{std::make_unique<std::uint64_t[]>(ToBytes(size))} {}

    inline void Set(std::size_t idx) noexcept
    {
        data_[idx >> shift] |= 1ull << (idx & lmask);
    }

    inline void Reset(std::size_t idx) noexcept
    {
        data_[idx >> shift] &= ~(1ull << (idx & lmask));
    }

    inline bool Test(std::size_t idx) const noexcept
    {
        return data_[idx >> shift] & (1ull << (idx & lmask));
    }

    inline std::size_t Size() const noexcept
    {
        return size_;
    }

    inline uint16_t Part(std::size_t idx) const noexcept
    {
        return data_[idx >> shift] >> (idx & lmask);
    }
private:
    inline std::size_t ToBytes(std::size_t val) const noexcept
    {
        return (val >> shift) + !!(val & lmask);
    }

    std::size_t                      size_, byte_size_;
    std::unique_ptr<std::uint64_t[]> data_;
};
