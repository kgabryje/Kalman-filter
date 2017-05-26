#pragma once

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace boost::numeric;
using namespace boost::numeric::ublas;

typedef matrix<double> mat;
typedef ublas::vector<double> vec;

class Kalman {
public:
    Kalman(mat A, mat H, mat Q, mat R);
    void predict(vec& X, mat& P);
    void correct(vec& X, const vec& Y, mat& P);
private:
    const mat _A;
    const mat _Q;
    const mat _R;
    const mat _H;
};
