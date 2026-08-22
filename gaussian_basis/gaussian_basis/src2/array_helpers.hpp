#include <vector>

#ifndef _ARRAY_HELPERS_
#define _ARRAY_HELPERS_

namespace array_helpers {

    class Array1D {
        unsigned int m_size;
        std::vector<double> m_data;
        public:
        Array1D(unsigned int size);
        Array1D(std::vector<double> v);
        double operator()(unsigned int) const;
        double &operator()(unsigned int);
        unsigned int size() const;
    };

    class Array2D;

    class SquareArray {
        unsigned int m_size;
        std::vector<double> m_data;
        public:
        SquareArray(unsigned int size);
        SquareArray(std::vector<std::vector<double>> m);
        double operator()(unsigned int i, unsigned int j) const;
        double &operator()(unsigned int i, unsigned int j);
        unsigned int row_size() const;
        SquareArray operator+(const SquareArray &m) const;
        SquareArray operator-(const SquareArray &m) const;
        double reduce(const Array2D &vectors) const;
        double reduce(const Array2D &arr1, const Array2D &arr2) const;
    };

    SquareArray operator*(double x, const SquareArray &arr);

    class Array2D {
        unsigned int m_row_size, m_col_size;
        std::vector<double> m_data;
        public:
        Array2D(unsigned int col_size, unsigned int row_size);
        Array2D(std::vector<std::vector<double>> m);
        double operator()(unsigned int i, unsigned int j) const;
        double &operator()(unsigned int i, unsigned int j);
        // TODO: create a View1D class or something.
        const double *c_ptr(unsigned int i) const;
        unsigned int row_size() const;
        unsigned int row_count() const;
        unsigned int col_size() const;
        unsigned int column_count() const;
    };

    Array2D row_stack(const Array2D &a, const Array2D &b);

    class SquareArraySlice;

    class HypercubeArray {
        unsigned int m_size;
        std::vector<double> m_data;
        public:
        HypercubeArray(unsigned int size);
        HypercubeArray(
            std::vector<
                std::vector<
                    std::vector<std::vector<double>>>>);
        double operator()(
            unsigned int a, unsigned int b,
            unsigned int c, unsigned int d) const;
        SquareArraySlice operator()(
            unsigned int a, unsigned int b);
        void operator()(
            unsigned int a, unsigned int b,
            SquareArraySlice slice
        );
        void operator()(
            unsigned int a, unsigned int b,
            const SquareArray &square_arr
        );
        double &operator()(
            unsigned int a, unsigned int b,
            unsigned int c, unsigned int d);
        unsigned int row_size() const;
        SquareArray reduce(
            unsigned int sum_index_a, unsigned int sum_index_b,
            const SquareArray &a, const SquareArray &b) const;
        SquareArray reduce(
            unsigned int sum_index_a, unsigned int sum_index_b,
            const Array2D &a, const Array2D &b) const;
        double reduce(
            int n_label1, int n_label2,
            const Array2D &n_arr1, const Array2D &n_arr2,
            int m_label1, int m_label2,
            const Array2D &m_arr1, const Array2D &m_arr2) const;
        double reduce(
            int n_label1, int n_label2,
            const Array1D &n_arr1, const Array1D &n_arr2,
            int m_label1, int m_label2,
            const Array1D &m_arr1, const Array1D &m_arr2) const;
        double reduce(
            int n_label1, int n_label2,
            const double *n_arr1, const double *n_arr2,
            int m_label1, int m_label2,
            const double *m_arr1, const double *m_arr2) const;
    };

    class SquareArraySlice {
        double *m_data_ptr;
        unsigned int m_size;
        public:
        SquareArraySlice(HypercubeArray &arr, int i, int j);
        double operator()(unsigned int i, unsigned int j) const;
        unsigned int row_size() const;
    };

    void test1();
    void test2();
    void test3();
    void test4();
    void test5();
    void test6();
    void test7();
    void test8();
    void test9();

};

#endif