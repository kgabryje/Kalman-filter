#pragma once

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

typedef boost::numeric::ublas::matrix<double> mat;
typedef boost::numeric::ublas::vector<double> vec;

class Kalman {
public:
    Kalman(mat A, mat H, mat P, mat Q, mat R);
    std::vector<double> estimate_position(const std::vector<double>& measured_position,
                                          const std::vector<double>& measured_velocity);
private:
    void predict(vec& X, mat& P);
    void correct(vec& X, const vec& Y, mat& P);
    void execute_step(vec& X, const vec& Y);

    mat error_covariance;
    const mat prediction_matrix;
    const mat process_noise_covariance;
    const mat sensor_noise_covariance;
    const mat observation_model;
};
