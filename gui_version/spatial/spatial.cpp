#include "spatial.hpp"

namespace spatial {

    Vector Vector::operator+(const struct Vector &z) const {
            return {{.t=this->t + z.t,
                    .x=this->x + z.x, 
                    .y=this->y + z.y, 
                    .z=this->z + z.z}};
        }

    Vector Vector::operator-(const struct Vector &v) const {
        return {{.t=this->t - v.t,
                .x=this->x - v.x, 
                .y=this->y - v.y, 
                .z=this->z - v.z}};
    }

    Vector Vector::operator*(double a) const {
        return {{.t=this->t*a, .x=this->x*a, .y=this->y*a, .z=this->z*a}};
    }

    Vector Vector::operator/(double a) const {
        return {{.t=this->t/a, .x=this->x/a, .y=this->y/a, .z=this->z/a}};
    }

    double Vector::operator[](int index) const {
        if (index < 4 && index > -1)
            return ind[index];
        return 0.0;
    };

    struct Vector operator+(double a, const struct Vector &v) {
        return {{.t=v.t + a,
                .x=v.x + a, 
                .y=v.y + a, 
                .z=v.z + a}};
    }

    struct Vector operator-(double a, const struct Vector &v) {
        return {{.t=a - v.t,
                .x=a - v.x, 
                .y=a - v.y, 
                .z=a - v.z}};
    }

    struct Vector operator*(double a, const struct Vector &v) {
        return {{.t=v.t*a,
                .x=v.x*a, 
                .y=v.y*a, 
                .z=v.z*a}};
    }

    struct Vector cross(const struct Vector &a, const struct Vector &b) {
        return {
            .t=0.0,
            .x=a[1]*b[2] - a[2]*b[1],
            .y=-a[0]*b[2] + a[2]*b[0],
            .z=a[0]*b[1] - a[1]*b[0]};
    }


    double dot(const struct Vector &a, const struct Vector &b) {
        return a.x*b.x + a.y*b.y + a.z*b.z;
    }

}
