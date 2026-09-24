#include "eigenvalues_eigenvectors.hpp"
// #include "Eigen/src/Core/util/Constants.h"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>

// #include <iostream>

using namespace Eigen;

void compute_eigenvalues_eigenvectors(
    array_helpers::Array1D &eigenvalues_array,
    array_helpers::Array2D &eigenvectors_array,
    const array_helpers::SquareArray &overlap_array,
    const array_helpers::SquareArray &matrix_array) {
    unsigned int matrix_size = matrix_array.row_size();
    unsigned int overlap_size = overlap_array.row_size();
    MatrixXd matrix 
        = Eigen::MatrixXd::Zero(matrix_size, matrix_size);
    for (int i = 0; i < matrix_size; i++)
        for (int j = 0; j < matrix_size; j++)
            matrix(i, j) = matrix_array(i, j);
    MatrixXd overlap
        = Eigen::MatrixXd::Zero(overlap_size, overlap_size);
    for (int i = 0; i < overlap_size; i++) {
        for (int j = 0; j < overlap_size; j++) {
            overlap(i, j) = overlap_array(i, j);
        }
    }
    GeneralizedSelfAdjointEigenSolver<MatrixXd> solver {};
    solver.compute(matrix, overlap);
    ComputationInfo info = solver.info();
    if (info == Eigen::NumericalIssue)
        return;
    MatrixXcd eigenvalues = solver.eigenvalues();
    for (int i = 0; i < eigenvalues.outerStride(); i++) {
        for (int j = 0; j < eigenvalues.innerStride(); j++) {
            if (i < eigenvalues_array.size())
                eigenvalues_array(i) = double(eigenvalues(i, 0).real());
            // std::cout << "Eigenvalue" << i << ", " << j << 
            // ": " << eigenvalues(i, j) << std::endl;
        }
    }
    MatrixXcd eigenvectors = solver.eigenvectors();
    // std::cout << eigenvectors.rows() << ", " 
    // << eigenvectors.cols() << std::endl;
    for (int i = 0; i < eigenvectors_array.col_size(); i++) {
        // There was a major bug that gave the incorrect energies.
        // It took a long time to realize that its cause was because
        // eigenvectors.innerStride was originally used here.
        // The outerStride method needs to be used instead since
        // this copies from the transpose of the eigenvectors matrix.
        for (int j = 0; j < eigenvectors.outerStride(); j++) {
            eigenvectors_array(i, j) = eigenvectors(j, i).real();
        }
    }

    
}
