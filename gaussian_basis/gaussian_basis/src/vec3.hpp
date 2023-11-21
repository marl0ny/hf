#ifndef _VEC3_
#define _VEC3_

struct Vec3 {
    union {
        double ind[3];
        union {
            struct {double x, y, z;};
            struct {double u, v, w;};
            struct {double r, g, b;};
            struct {double i, j, k;};
        };
    };
    Vec3 operator+(const struct Vec3 &z) const;
    Vec3 operator-(const struct Vec3 &z) const;
    Vec3 operator*(double a) const;
    Vec3 operator/(double a) const;
    double operator[](int index) const;
};

struct Vec3 operator+(double x, const struct Vec3 &v);

struct Vec3 operator-(double x, const struct Vec3 &v);

struct Vec3 operator*(double x, const struct Vec3 &v);

double dot(const struct Vec3 &a, const struct Vec3 &b);

#endif
