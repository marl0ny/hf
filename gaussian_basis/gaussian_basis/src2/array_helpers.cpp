#include "array_helpers.hpp"

#include <iostream>

namespace array_helpers {

Array1D::Array1D(unsigned int size) {
    m_size = size;
    m_data = std::vector(size, 0.0);
}

Array1D::Array1D(std::vector<double> v) {
    m_data = std::vector(v);
}

double Array1D::operator()(unsigned int i) const {
    return this->m_data[i];
}

double &Array1D::operator()(unsigned int i) {
    return this->m_data[i];
}

unsigned int Array1D::size() const {
    return this->m_size;
}

SquareArray::SquareArray(unsigned int size) {
    this->m_size = size;
    this->m_data = std::vector<double>(size*size, 0.0);
}

SquareArray::SquareArray(std::vector<std::vector<double>> m) {
    unsigned int size0 = 0;
    if (m.size() > 0)
        size0 = m[0].size();
    unsigned int size1 = m.size();
    unsigned int size = std::min(size0, size1);
    this->m_size = size;
    this->m_data = std::vector<double>(size*size, 0.0);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
            this->operator()(i, j) = m[i][j];
    }
}

double SquareArray::operator()(unsigned int i, unsigned int j) const {
    if (this->row_size() == 0)
        return 0.0;
    return m_data[i*m_size + j];
}

double &SquareArray::operator()(unsigned int i, unsigned int j) {
    return m_data[i*m_size + j];
}

unsigned int SquareArray::row_size() const {
    return m_size;
}

SquareArray SquareArray::operator+(const SquareArray &m) const {
    SquareArray arr2(this->row_size());
    for (int i = 0; i < std::min(row_size(), m.row_size()); i++)
        for (int j = 0; j < std::min(row_size(), m.row_size()); j++)
            arr2(i, j) = m(i, j) + this->operator()(i, j); 
    return arr2;
}

SquareArray SquareArray::operator-(const SquareArray &m) const {
    SquareArray arr2(this->row_size());
    for (int i = 0; i < std::min(row_size(), m.row_size()); i++)
        for (int j = 0; j < std::min(row_size(), m.row_size()); j++)
            arr2(i, j) = this->operator()(i, j) - m(i, j); 
    return arr2;
}

double SquareArray::reduce(const Array2D &m) const {
    // if (m.row_size() != this->row_size())
    //     return 0.0;
    double sum = 0.0;
    for (int m_ind = 0; m_ind < m.row_count(); m_ind++) {
        for (int i = 0; i < this->row_size(); i++) {
            for (int j = 0; j < this->row_size(); j++) {
                sum += m(m_ind, j)*m(m_ind, i)*this->operator()(i, j);
            }
        }
    }
    return sum;
}

double SquareArray::reduce(const Array2D &m1, const Array2D &m2) const {
    if (m1.row_size() != m2.row_size())
        return 0.0;
    if (m1.row_size() != this->row_size())
        return 0.0;
    double sum = 0.0;
    for (int m_ind = 0; m_ind < m1.col_size(); m_ind++) {
        for (int i = 0; i < this->row_size(); i++) {
            for (int j = 0; j < this->row_size(); j++) {
                sum += m1(m_ind, i)*m2(m_ind, j)*this->operator()(i, j);
            }
        }
    }
    return sum;
}

SquareArray operator*(double x, const SquareArray &arr) {
    SquareArray arr2(arr.row_size());
    for (int i = 0; i < arr.row_size(); i++)
        for (int j = 0; j < arr.row_size(); j++)
            arr2(i, j) = x*arr(i, j);
    return arr2;
}

Array2D::Array2D(unsigned int col_size, unsigned int row_size) {
    this->m_col_size = col_size;
    this->m_row_size = row_size;
    this->m_data = std::vector<double>(col_size*row_size, 0.0);
}

Array2D::Array2D(std::vector<std::vector<double>> m) {
    if (m.size() > 0)
        m_row_size = m[0].size();
    m_col_size = m.size();
    this->m_data = std::vector<double>(m_col_size*m_row_size, 0.0);
    for (int i = 0; i < m_col_size; i++) {
        for (int j = 0; j < m_row_size; j++)
            this->operator()(i, j) = m[i][j];
    }
}

double Array2D::operator()(unsigned int i, unsigned int j) const {
    if (i >= m_col_size || j >= m_row_size)
        return 0.0;
    return this->m_data[i*m_row_size + j];
}

double &Array2D::operator()(unsigned int i, unsigned int j) {
    // if (i >= m_col_size || j >= m_row_size)
    //     return;
    return this->m_data[i*m_row_size + j];
}

const double *Array2D::c_ptr(unsigned int i) const {
    return (const double *)&this->m_data[i*m_row_size];
}

unsigned int Array2D::row_size() const {
    return this->m_row_size;
}

unsigned int Array2D::row_count() const {
    return this->m_col_size;
}

unsigned int Array2D::col_size() const {
    return this->m_col_size;
}

unsigned int Array2D::column_count() const {
    return this->m_row_size;
}

Array2D row_stack(const Array2D &a, const Array2D &b) {
    int number_of_columns = std::min(a.column_count(), b.column_count());
    int number_of_rows = a.row_count() + b.row_count();
    Array2D arr(number_of_rows, number_of_columns);
    for (int i = 0; i < a.col_size(); i++)
        for (int j = 0; j < number_of_columns; j++)
            arr(i, j) = a(i, j);
    for (int i = 0; i < b.col_size(); i++)
        for (int j = 0; j < number_of_columns; j++)
            arr(i + a.col_size(), j) = b(i, j);
    return arr;
}

HypercubeArray::HypercubeArray(unsigned int size) {
    this->m_size = size;
    this->m_data = std::vector<double>(size*size*size*size, 0.0);
}

HypercubeArray
::HypercubeArray(
    std::vector<std::vector<std::vector<std::vector<double>>>> a) {
    unsigned int size0 = 0, size1 = 0, size2 = 0, size3 = 0;
    // TODO: check each subrows are of the same size.
    if (a.size() > 0 
        && a[0].size() > 0 
        && a[0][0].size() > 0 
        && a[0][0][0].size() > 0)
        size3 = a[0][0][0].size();
    if (a.size() > 0 
        && a[0].size() > 0 
        && a[0][0].size() > 0)
        size2 = a[0][0].size();
    if (a.size() > 0 
        && a[0].size() > 0)
        size1 = a[0].size();
    if (a.size() > 0)
        size0 = a.size();
    int size = std::min(std::min(std::min(size0, size1), size2), size3);
    for (int a_ = 0; a_ < size; a_++)
        for (int b = 0; b < size; b++)
            for (int c = 0; c < size; c++)
                for (int d = 0; d < size; d++)
                    this->operator()(a_, b, c, d)
                        = a[a_][b][c][d];

}

SquareArraySlice HypercubeArray::operator()(
    unsigned int a, unsigned int b) {
    return SquareArraySlice(*this, a, b);
}

double HypercubeArray::operator()(
    unsigned int a, unsigned int b, 
    unsigned int c, unsigned int d) const {
    if (this->row_size() == 0)
        return 0.0;
    unsigned int n = m_size;
    return m_data[a*n*n*n + b*n*n + c*n + d];
}

void HypercubeArray::operator()(
    unsigned int i, unsigned int j, 
    SquareArraySlice slice) {
    unsigned int n = m_size;
    for (int a = 0; a < std::min(slice.row_size(), n); a++)
        for (int b = 0; b < std::min(slice.row_size(), n); b++)
            this->operator()(i, j, a, b) = slice(a, b);
}

void HypercubeArray::operator()(
    unsigned int i, unsigned int j,
    const SquareArray &square_arr) {
    unsigned int n = m_size;
    for (int a = 0; a < std::min(square_arr.row_size(), n); a++)
        for (int b = 0; b < std::min(square_arr.row_size(), n); b++)
            this->operator()(i, j, a, b) = square_arr(a, b);
}

double &HypercubeArray::operator()(
    unsigned int a, unsigned int b, 
    unsigned int c, unsigned int d){
    unsigned int n = m_size;
    return m_data[a*n*n*n + b*n*n + c*n + d];
}

SquareArray HypercubeArray::reduce(
        unsigned int sum_label_a, unsigned int sum_label_b,
        const SquareArray &a, const SquareArray &b) const {
    SquareArray res(this->row_size());
    sum_label_a = sum_label_a % 4;
    sum_label_b = sum_label_b % 4;
    if (sum_label_a == sum_label_b)
        return res;
    unsigned int final_labels[2] = {0, 0};
    unsigned int final_labels_index = 0;
    for (int indices_label = 0; indices_label < 4; indices_label++)
        if (indices_label != sum_label_a && indices_label != sum_label_b)
            final_labels[final_labels_index++] = indices_label;
    // std::cout << "Sum label a: " << sum_label_a << std::endl;
    // std::cout << "Sum label b: " << sum_label_b << std::endl;
    // std::cout << "Final label 0: " << final_labels[0] << std::endl;
    // std::cout << "Final label 1: " << final_labels[1] << std::endl;
    for (int ind0 = 0; ind0 < this->row_size(); ind0++) {
        for (int ind1 = 0; ind1 < this->row_size(); ind1++) {
            for (int m_ind = 0; m_ind < this->row_size(); m_ind++) {
                for (int ind2 = 0; ind2 < this->row_size(); ind2++) {
                    for (int ind3 = 0; ind3 < this->row_size(); ind3++) {
                        int sum_index_a = 0, sum_index_b = 1;
                        int final_index0 = 0, final_index1 = 1;
                        unsigned int indices[4] = {0, 0, 0, 0};
                        switch (sum_label_a) {
                            case 0:
                            sum_index_a = ind0;
                            break;
                            case 1:
                            sum_index_a = ind1;
                            break;
                            case 2:
                            sum_index_a = ind2;
                            break;
                            case 3:
                            sum_index_a = ind3;
                            break;
                            default:
                            break;
                        }
                        switch (sum_label_b) {
                            case 0:
                            sum_index_b = ind0;
                            break;
                            case 1:
                            sum_index_b = ind1;
                            break;
                            case 2:
                            sum_index_b = ind2;
                            break;
                            case 3:
                            sum_index_b = ind3;
                            break;
                            default:
                            break;
                        }
                        switch (final_labels[0]) {
                            case 0:
                            final_index0 = ind0;
                            break;
                            case 1:
                            final_index0 = ind1;
                            break;
                            case 2:
                            final_index0 = ind2;
                            break;
                            case 3:
                            final_index0 = ind3;
                            break;
                            default:
                            break;
                        }
                        switch (final_labels[1]) {
                            case 0:
                            final_index1 = ind0;
                            break;
                            case 1:
                            final_index1 = ind1;
                            break;
                            case 2:
                            final_index1 = ind2;
                            break;
                            case 3:
                            final_index1 = ind3;
                            break;
                            default:
                            break;
                        }
                        indices[sum_label_a] = sum_index_a;
                        indices[sum_label_b] = sum_index_b;
                        indices[final_labels[0]] = final_index0;
                        indices[final_labels[1]] = final_index1;
                        res(final_index0, final_index1)
                            += this->operator()(
                                indices[0], indices[1], 
                                indices[2], indices[3])
                                *a(m_ind, sum_index_a)*b(m_ind, sum_index_b);
                    }

                }
            }
        }
    }
    return res;
}

SquareArray HypercubeArray::reduce(
        unsigned int sum_label_a, unsigned int sum_label_b,
        const Array2D &a, const Array2D &b) const {
    SquareArray res(this->row_size());
    sum_label_a = sum_label_a % 4;
    sum_label_b = sum_label_b % 4;
    if (sum_label_a == sum_label_b)
        return res;
    if (a.row_size() != b.row_size() &&  a.row_size() != this->row_size())
        return res;
    unsigned int mat_col_size = a.col_size();
    unsigned int final_labels[2] = {0, 0};
    unsigned int final_labels_index = 0;
    for (int indices_label = 0; indices_label < 4; indices_label++)
        if (indices_label != sum_label_a && indices_label != sum_label_b)
            final_labels[final_labels_index++] = indices_label;
    // std::cout << "Sum label a: " << sum_label_a << std::endl;
    // std::cout << "Sum label b: " << sum_label_b << std::endl;
    // std::cout << "Final label 0: " << final_labels[0] << std::endl;
    // std::cout << "Final label 1: " << final_labels[1] << std::endl;
    for (int m_ind = 0; m_ind < mat_col_size; m_ind++) {
        for (int ind0 = 0; ind0 < this->row_size(); ind0++) {
            for (int ind1 = 0; ind1 < this->row_size(); ind1++) {
                for (int ind2 = 0; ind2 < this->row_size(); ind2++) {
                    for (int ind3 = 0; ind3 < this->row_size(); ind3++) {
                        int sum_index_a = 0, sum_index_b = 0;
                        int final_index0 = 0, final_index1 = 0;
                        unsigned int indices[4] = {0, 0, 0, 0};
                        switch (sum_label_a) {
                            case 0: sum_index_a = ind0;
                            break;
                            case 1: sum_index_a = ind1;
                            break;
                            case 2: sum_index_a = ind2;
                            break;
                            case 3: sum_index_a = ind3;
                            break;
                            default: break;
                        }
                        switch (sum_label_b) {
                            case 0: sum_index_b = ind0;
                            break;
                            case 1: sum_index_b = ind1;
                            break;
                            case 2: sum_index_b = ind2;
                            break;
                            case 3: sum_index_b = ind3;
                            break;
                            default: break;
                        }
                        switch (final_labels[0]) {
                            case 0: final_index0 = ind0;
                            break;
                            case 1: final_index0 = ind1;
                            break;
                            case 2: final_index0 = ind2;
                            break;
                            case 3: final_index0 = ind3;
                            break;
                            default: break;
                        }
                        switch (final_labels[1]) {
                            case 0: final_index1 = ind0;
                            break;
                            case 1: final_index1 = ind1;
                            break;
                            case 2: final_index1 = ind2;
                            break;
                            case 3: final_index1 = ind3;
                            break;
                            default: break;
                        }
                        indices[sum_label_a] = sum_index_a;
                        indices[sum_label_b] = sum_index_b;
                        indices[final_labels[0]] = final_index0;
                        indices[final_labels[1]] = final_index1;
                        res(final_index0, final_index1)
                            += this->operator()(
                                indices[0], indices[1], 
                                indices[2], indices[3])
                                *a(m_ind, sum_index_a)*b(m_ind, sum_index_b);
                    }

                }
            }
        }
    }
    return res;
}

double 
HypercubeArray::reduce(
    int n_label1, int n_label2,
    const Array2D &n_arr1, const Array2D &n_arr2,
    int m_label1, int m_label2,
    const Array2D &m_arr1, const Array2D &m_arr2) const {
    if (n_arr1.col_size() != n_arr2.col_size())
        return 0.0;
    if (n_arr1.row_size() != this->row_size() || 
        n_arr1.row_size() != n_arr2.row_size())
        return 0.0;
    if (m_arr1.col_size() != m_arr2.col_size())
        return 0.0;
    if (m_arr1.row_size() != this->row_size() || 
        m_arr1.row_size() != m_arr2.row_size())
        return 0.0;
    unsigned int n_size = n_arr1.col_size();
    unsigned int m_size = m_arr1.col_size();
    double sum = 0.0;
    for (int m = 0; m < m_size; m++) {
        for (int n = 0; n < n_size; n++) {
            for (int i = 0; i < this->row_size(); i++) {
                for (int j = 0; j < this->row_size(); j++) {
                    for (int k = 0; k < this->row_size(); k++) {
                        for (int l = 0; l < this->row_size(); l++) {
                            int m_index1 = (m_label1 == 0)? i: 
                                ((m_label1 == 1)? j: ((m_label1 == 2)? k: l));
                            int m_index2 = (m_label2 == 0)? i: 
                                ((m_label2 == 1)? j: ((m_label2 == 2)? k: l));
                            int n_index1 = (n_label1 == 0)? i: 
                                ((n_label1 == 1)? j: ((n_label1 == 2)? k: l));
                            int n_index2 = (n_label2 == 0)? i: 
                                ((n_label2 == 1)? j: ((n_label2 == 2)? k: l));
                            sum += this->operator()(i, j, k, l)
                                *m_arr1(m, m_index1)*m_arr2(m, m_index2)
                                *n_arr1(n, n_index1)*n_arr2(n, n_index2);
                        }
                    }
                }
            }
        }
    }
    return sum;

}

double HypercubeArray::reduce(
    int n_label1, int n_label2,
    const Array1D &n_arr1, const Array1D &n_arr2,
    int m_label1, int m_label2,
    const Array1D &m_arr1, const Array1D &m_arr2) const {
    if (n_arr1.size() != this->row_size() || 
        n_arr1.size() != n_arr2.size())
        return 0.0;
    if (m_arr1.size() != this->row_size() || 
        m_arr1.size() != m_arr2.size())
        return 0.0;
    double sum = 0.0;
    for (int i = 0; i < this->row_size(); i++) {
        for (int j = 0; j < this->row_size(); j++) {
            for (int k = 0; k < this->row_size(); k++) {
                for (int l = 0; l < this->row_size(); l++) {
                    int m_index1 = (m_label1 == 0)? i: 
                        ((m_label1 == 1)? j: ((m_label1 == 2)? k: l));
                    int m_index2 = (m_label2 == 0)? i: 
                        ((m_label2 == 1)? j: ((m_label2 == 2)? k: l));
                    int n_index1 = (n_label1 == 0)? i: 
                        ((n_label1 == 1)? j: ((n_label1 == 2)? k: l));
                    int n_index2 = (n_label2 == 0)? i: 
                        ((n_label2 == 1)? j: ((n_label2 == 2)? k: l));
                    sum += this->operator()(i, j, k, l)
                        *m_arr1(m_index1)*m_arr2(m_index2)
                        *n_arr1(n_index1)*n_arr2(n_index2);
                }
            }
        }
    }
    return sum;
}

double HypercubeArray::reduce(
    int n_label1, int n_label2,
    const double *n_arr1, const double *n_arr2,
    int m_label1, int m_label2,
    const double *m_arr1, const double *m_arr2) const {
    double sum = 0.0;
    for (int i = 0; i < this->row_size(); i++) {
        for (int j = 0; j < this->row_size(); j++) {
            for (int k = 0; k < this->row_size(); k++) {
                for (int l = 0; l < this->row_size(); l++) {
                    int m_index1 = (m_label1 == 0)? i: 
                        ((m_label1 == 1)? j: ((m_label1 == 2)? k: l));
                    int m_index2 = (m_label2 == 0)? i: 
                        ((m_label2 == 1)? j: ((m_label2 == 2)? k: l));
                    int n_index1 = (n_label1 == 0)? i: 
                        ((n_label1 == 1)? j: ((n_label1 == 2)? k: l));
                    int n_index2 = (n_label2 == 0)? i: 
                        ((n_label2 == 1)? j: ((n_label2 == 2)? k: l));
                    sum += this->operator()(i, j, k, l)
                        *m_arr1[m_index1]*m_arr2[m_index2]
                        *n_arr1[n_index1]*n_arr2[n_index2];
                }
            }
        }
    }
    return sum;
}

unsigned int HypercubeArray::row_size() const {
    return m_size;
}

SquareArraySlice::SquareArraySlice(HypercubeArray &arr, int i, int j) {
    this->m_size = arr.row_size();
    m_data_ptr = &arr(i, j, 0, 0);
}

double SquareArraySlice::operator()(unsigned int i, unsigned int j) const {
    return m_data_ptr[i*m_size + j];
}

unsigned int SquareArraySlice::row_size() const {
    return m_size;
}

void test1() {
    HypercubeArray arr(3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    arr(i, j, k, l) = i + j + k + l;
                }
            }
        }
    }
    SquareArray a(
        {
            {1, 2, 3}, 
            {4, 5, 6}, 
            {7, 8, 9}}
    );
    SquareArray b(
        {
            {1, -1, 1}, 
            {1,0, 1}, 
            {1, 1, 1}}
    );
    SquareArray res = arr.reduce(2, 3, a, b);
    // for (int i = 0; i < 3; i++) {
    //     for (int j = 0; j < 3; j++)
    //         std::cout << i << ", " << j << ": " << res(i, j) << std::endl;
    // }
    SquareArray answer = SquareArray(
        {{228.0, 336.0, 444.0},
         {336.0, 444.0, 552.0}, 
         {444.0, 552.0, 660.0}});
    std::vector<std::vector<int>> indices {};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (std::abs(answer(i, j) - res(i, j)) > 1e-40) {
                indices.push_back({i, j});
            }
        }
    }
    if (indices.size() > 0) {
        std::cout << "Test case failed: " << std::endl;
        for (auto &index: indices) {
            std::cout << "Expected " << answer(index[0], index[1])
                << " at (" << index[0] << ", " << index[1] << ").\n";
            std::cout << "Got " 
                << res(index[0], index[1]) << " instead.\n";
        }
    }

}

void test2() {
    HypercubeArray arr(2);
    arr(0, 0, SquareArray({{1, 2}, {3, 4}}));
    arr(0, 1, SquareArray({{5, 6}, {7, 8}}));
    arr(1, 0, SquareArray({{-4, -3}, {-2, -1}}));
    arr(1, 1, SquareArray({{1, 1}, {1, 1}}));
    SquareArray a({{1, 2}, {4, 5}});
    SquareArray b({{1, -1}, {1,0}}); 
    SquareArray res = arr.reduce(2, 3, a, b);

    // for (int i = 0; i < 2; i++) {
    //     for (int j = 0; j < 2; j++)
    //         std::cout << i << ", " << j << ": " << res(i, j) << std::endl;
    // }

    SquareArray answer = SquareArray(
        {{16.0, 52.0}, {-29.0, 9.0}});
    std::vector<std::vector<int>> indices {};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (std::abs(answer(i, j) - res(i, j)) > 1e-40) {
                indices.push_back({i, j});
            }
        }
    }
    if (indices.size() > 0) {
        std::cout << "Test case failed: " << std::endl;
        for (auto &index: indices) {
            std::cout << "Expected " << answer(index[0], index[1])
                << " at (" << index[0] << ", " << index[1] << ").\n";
            std::cout << "Got " 
                << res(index[0], index[1]) << " instead.\n";
        }
    }
}

void test3() {
    HypercubeArray arr(2);
    arr(0, 0, SquareArray({{1, 2}, {3, 4}}));
    arr(0, 1, SquareArray({{5, 6}, {7, 8}}));
    arr(1, 0, SquareArray({{-4, -3}, {-2, -1}}));
    arr(1, 1, SquareArray({{1, 1}, {1, 1}}));
    SquareArray a({{1, 2}, {4, 5}});
    SquareArray b({{1, -1}, {1,0}}); 
    SquareArray res = arr.reduce(1, 3, a, b);

    // for (int i = 0; i < 2; i++) {
    //     for (int j = 0; j < 2; j++)
    //         std::cout << i << ", " << j << ": " << res(i, j) << std::endl;
    // }

    SquareArray answer = SquareArray(
        {{26.0, 44.0}, {-12.0, -4.0}});
    std::vector<std::vector<int>> indices {};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (std::abs(answer(i, j) - res(i, j)) > 1e-40) {
                indices.push_back({i, j});
            }
        }
    }
    if (indices.size() > 0) {
        std::cout << "Test case failed: " << std::endl;
        for (auto &index: indices) {
            std::cout << "Expected " << answer(index[0], index[1])
                << " at (" << index[0] << ", " << index[1] << ").\n";
            std::cout << "Got " 
                << res(index[0], index[1]) << " instead.\n";
        }
    }
}

void test4() {
    HypercubeArray arr(2);
    arr(0, 0, SquareArray({{1, 2}, {3, 4}}));
    arr(0, 1, SquareArray({{5, 6}, {7, 8}}));
    arr(1, 0, SquareArray({{-4, -3}, {-2, -1}}));
    arr(1, 1, SquareArray({{1, 1}, {1, 1}}));
    SquareArray a({{1, 2}, {4, 5}});
    SquareArray b({{1, -1}, {1,0}}); 
    SquareArray res = arr.reduce(0, 3, a, b);

    // for (int i = 0; i < 2; i++) {
    //     for (int j = 0; j < 2; j++)
    //         std::cout << i << ", " << j << ": " << res(i, j) << std::endl;
    // }

    SquareArray answer = SquareArray(
        {{-19.0, -1.0}, {24.0, 32.0}});
    std::vector<std::vector<int>> indices {};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (std::abs(answer(i, j) - res(i, j)) > 1e-40) {
                indices.push_back({i, j});
            }
        }
    }
    if (indices.size() > 0) {
        std::cout << "Test case failed: " << std::endl;
        for (auto &index: indices) {
            std::cout << "Expected " << answer(index[0], index[1])
                << " at (" << index[0] << ", " << index[1] << ").\n";
            std::cout << "Got " 
                << res(index[0], index[1]) << " instead.\n";
        }
    }
}

void test5() {
    HypercubeArray arr(2);
    arr(0, 0, SquareArray({{1, 2}, {3, 4}}));
    arr(0, 1, SquareArray({{5, 6}, {7, 8}}));
    arr(1, 0, SquareArray({{-4, -3}, {-2, -1}}));
    arr(1, 1, SquareArray({{1, 1}, {1, 1}}));
    SquareArray a({{1, 2}, {4, 5}});
    SquareArray b({{1, -1}, {1,0}}); 
    SquareArray res = arr.reduce(2, 0, a, b);

    // for (int i = 0; i < 2; i++) {
    //     for (int j = 0; j < 2; j++)
    //         std::cout << i << ", " << j << ": " << res(i, j) << std::endl;
    // }

    SquareArray answer = SquareArray(
        {{34.0, 43.0}, {71.0, 83.0}});
    std::vector<std::vector<int>> indices {};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (std::abs(answer(i, j) - res(i, j)) > 1e-40) {
                indices.push_back({i, j});
            }
        }
    }
    if (indices.size() > 0) {
        std::cout << "Test case failed: " << std::endl;
        for (auto &index: indices) {
            std::cout << "Expected " << answer(index[0], index[1])
                << " at (" << index[0] << ", " << index[1] << ").\n";
            std::cout << "Got " 
                << res(index[0], index[1]) << " instead.\n";
        }
    }
}

void test6() {
    HypercubeArray arr(2);
    arr(0, 0, SquareArray({{1, 2}, {3, 4}}));
    arr(0, 1, SquareArray({{5, 6}, {7, 8}}));
    arr(1, 0, SquareArray({{-4, -3}, {-2, -1}}));
    arr(1, 1, SquareArray({{1, 1}, {1, 1}}));
    arr(0 , 0, arr(1, 1));
    arr(0 , 1, arr(1, 0));
    HypercubeArray answer(2);
    answer(0, 0, SquareArray({{1, 1}, {1, 1}}));
    answer(0, 1, SquareArray({{-4, -3}, {-2, -1}}));
    answer(1, 0, SquareArray({{-4, -3}, {-2, -1}}));
    answer(1, 1, SquareArray({{1, 1}, {1, 1}}));
    std::vector<std::vector<int>> indices {};
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                for (int l_ = 0; l_ < 2; l_++)
                    if (std::abs(answer(i, j, k, l_) 
                            - arr(i, j, k, l_)) > 1e-40)
                        indices.push_back({i, j, k, l_});
    if (indices.size() > 0) {
        std::cout << "Test case failed: " << std::endl;
        for (auto &index: indices) {
            std::cout << "Expected " << answer(
                index[0], index[1], index[2], index[3])
                << " at (" << index[0] << ", " << index[1] << ").\n";
            std::cout << "Got " 
                << arr(index[0], index[1], index[2], index[3]) << " instead.\n";
        }
    }
}

void test7() {
    HypercubeArray arr(2);
    arr(0, 0, SquareArray({{1, 2}, {3, 4}}));
    arr(0, 1, SquareArray({{5, 6}, {7, 8}}));
    arr(1, 0, SquareArray({{-4, -3}, {-2, -1}}));
    arr(1, 1, SquareArray({{1, 1}, {1, 1}}));
    Array2D a({{1, 2}, {4, 5}, {1, 1}});
    Array2D b({{1, -1}, {1,0}, {1, -2}});
    SquareArray res = arr.reduce(2, 0, a, b);

    // for (int i = 0; i < 2; i++) {
    //     for (int j = 0; j < 2; j++)
    //         std::cout << i << ", " << j << ": " << res(i, j) << std::endl;
    // }

    SquareArray answer = SquareArray(
        {{50.0, 57.0}, {79.0, 93.0}});
    std::vector<std::vector<int>> indices {};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (std::abs(answer(i, j) - res(i, j)) > 1e-40) {
                indices.push_back({i, j});
            }
        }
    }
    if (indices.size() > 0) {
        std::cout << "Test case failed: " << std::endl;
        for (auto &index: indices) {
            std::cout << "Expected " << answer(index[0], index[1])
                << " at (" << index[0] << ", " << index[1] << ").\n";
            std::cout << "Got " 
                << res(index[0], index[1]) << " instead.\n";
        }
    }
}

void test8() {
    HypercubeArray arr(2);
    arr(0, 0, SquareArray({{1, 2}, {3, 4}}));
    arr(0, 1, SquareArray({{5, 6}, {7, 8}}));
    arr(1, 0, SquareArray({{-4, -3}, {-2, -1}}));
    arr(1, 1, SquareArray({{1, 1}, {1, 1}}));
    Array2D a({
        {1, 2}, 
        {4, 5}, 
        {1, 1},
        {0, -3}
    });
    Array2D b({
        {1, -1}, 
        {1,0}, 
        {1, -2},
        {0, 1}
    });
    SquareArray res = arr.reduce(2, 3, a, b);

    // for (int i = 0; i < 2; i++) {
    //     for (int j = 0; j < 2; j++)
    //         std::cout << i << ", " << j << ": " << res(i, j) << std::endl;
    // }

    SquareArray answer = SquareArray(
        {{-4.0, 12.0}, {-24.0, 4.0}});
    std::vector<std::vector<int>> indices {};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (std::abs(answer(i, j) - res(i, j)) > 1e-40) {
                indices.push_back({i, j});
            }
        }
    }
    if (indices.size() > 0) {
        std::cout << "Test case failed: " << std::endl;
        for (auto &index: indices) {
            std::cout << "Expected " << answer(index[0], index[1])
                << " at (" << index[0] << ", " << index[1] << ").\n";
            std::cout << "Got " 
                << res(index[0], index[1]) << " instead.\n";
        }
    }
}

void test9() {
    HypercubeArray arr(2);
    arr(0, 0, SquareArray({{1, 2}, {3, 4}}));
    arr(0, 1, SquareArray({{5, 6}, {7, 8}}));
    arr(1, 0, SquareArray({{-4, -3}, {-2, -1}}));
    arr(1, 1, SquareArray({{1, 1}, {1, 1}}));
    Array2D a({
        {1, 2}, 
        {4, 5}, 
        {1, 1},
        {0, -3}
    });
    Array2D b({
        {1, -1}, 
        {1,0}, 
        {1, -2},
        {0, 1}
    });
    SquareArray res = arr.reduce(1, 3, a, b);

    // for (int i = 0; i < 2; i++) {
    //     for (int j = 0; j < 2; j++)
    //         std::cout << i << ", " << j << ": " << res(i, j) << std::endl;
    // }

    SquareArray answer = SquareArray(
        {{-2.0, 6.0}, {-14.0, -8.0}});
    std::vector<std::vector<int>> indices {};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (std::abs(answer(i, j) - res(i, j)) > 1e-40) {
                indices.push_back({i, j});
            }
        }
    }
    if (indices.size() > 0) {
        std::cout << "Test case failed: " << std::endl;
        for (auto &index: indices) {
            std::cout << "Expected " << answer(index[0], index[1])
                << " at (" << index[0] << ", " << index[1] << ").\n";
            std::cout << "Got " 
                << res(index[0], index[1]) << " instead.\n";
        }
    }
}

}