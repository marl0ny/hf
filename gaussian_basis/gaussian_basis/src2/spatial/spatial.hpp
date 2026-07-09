#include <cstdint>

#ifndef _SPATIAL_
#define _SPATIAL_

namespace spatial {

    struct Vector {
        union {
            struct {double t, x, y, z; };
            struct {double ind[4]; };
        };
        Vector operator+(const struct Vector &z) const;
        Vector operator-(const struct Vector &z) const;
        Vector operator*(double a) const;
        Vector operator/(double a) const;
        double operator[](int index) const;
    };

    struct Vector operator+(double x, const struct Vector &v);

    struct Vector operator-(double x, const struct Vector &v);

    struct Vector operator*(double x, const struct Vector &v);

    struct Vector cross(const struct Vector &a, const struct Vector &b);

    double dot(const struct Vector &a, const struct Vector &b);

    struct UByte4 {
        union {
            struct {uint8_t x, y, z, w; };
            struct {uint8_t i, j, k, l; };
            struct {uint8_t ind[4]; };
        };
    };

}

#endif