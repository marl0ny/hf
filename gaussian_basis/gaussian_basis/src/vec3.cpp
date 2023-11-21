#include "vec3.hpp"

Vec3 Vec3::operator+(const struct Vec3 &z) const {
        return {{.x=this->x + z.x, 
                 .y=this->y + z.y, 
                 .z=this->z + z.z}};
    }

Vec3 Vec3::operator-(const struct Vec3 &v) const {
    return {{.x=this->x - v.x, 
             .y=this->y - v.y, 
             .z=this->z - v.z}};
}

Vec3 Vec3::operator*(double a) const {
    return {{.x=this->x*a, .y=this->y*a, .z=this->z*a}};
}

Vec3 Vec3::operator/(double a) const {
    return {{.x=this->x/a, .y=this->y/a, .z=this->z/a}};
}

double Vec3::operator[](int index) const {
    if (index < 3 && index > -1)
        return ind[index];
    return 0.0;
};

struct Vec3 operator+(double a, const struct Vec3 &v) {
    return {{.x=v.x + a, 
             .y=v.y + a, 
             .z=v.z + a}};
}

struct Vec3 operator-(double a, const struct Vec3 &v) {
    return {{.x=a - v.x, 
             .y=a - v.y, 
             .z=a - v.z}};
}

struct Vec3 operator*(double a, const struct Vec3 &v) {
    return {{.x=v.x*a, 
             .y=v.y*a, 
             .z=v.z*a}};
}


double dot(const struct Vec3 &a, const struct Vec3 &b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
