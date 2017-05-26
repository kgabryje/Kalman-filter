#pragma once

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

class Kalman {
    typedef boost::numeric::ublas::matrix<double> mat;
    typedef boost::numeric::ublas::vector<double> vec;
public:
    Kalman(mat prediction_matrix, mat observation_model, mat initial_error_covariance, mat process_noise_covariance,
           mat sensor_noise_covariance);
    std::vector<double> estimate_position(const std::vector<double>& measured_position,
                                          const std::vector<double>& measured_velocity);
private:
    void predict(vec& X, mat& P);
    void correct(vec& X, const vec& Y, mat& P);
    void execute_step(vec& X, const vec& Y, mat& P);

    const mat initial_error_covariance;
    const mat prediction_matrix;
    const mat process_noise_covariance;
    const mat sensor_noise_covariance;
    const mat observation_model;
};
