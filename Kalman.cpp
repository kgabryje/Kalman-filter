#include "Kalman.h"

mat invert_matrix(const mat& input_matrix) {
    mat A(input_matrix);
    mat output_matrix(A.size1(), A.size1());
    permutation_matrix<std::size_t> pm(A.size1());
    lu_factorize(A, pm);
    output_matrix.assign(identity_matrix<double> (A.size1()));
    lu_substitute(A, pm, output_matrix);
    return output_matrix;
}

Kalman::Kalman(mat A, mat H, mat Q, mat R) : _A(A),  _H(H), _Q(Q), _R(R) {};

void Kalman::predict(vec& X, mat& P) {
    X = prod(_A, X);
    P = prod(mat(prod(_A, P)), trans(_A)) + _Q;
}

void Kalman::correct(vec& X, const vec& Y, mat& P) {
    mat HPH = prod(mat(prod(_H, P)), trans(_H));
    mat inv = invert_matrix(HPH + _R);
    mat K = prod(mat(prod(P, trans(_H))), inv);
    X = X + prod(K, vec(Y - prod(_H, X)));
    P = prod((identity_matrix<double> (P.size1()) - prod(K, _H)), P);
}