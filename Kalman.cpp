#include "Kalman.h"

using namespace boost::numeric::ublas;

matrix<double> invert_matrix(const matrix<double>& input_matrix) {
    matrix<double> A(input_matrix);
    matrix<double> output_matrix(A.size1(), A.size1());
    permutation_matrix<std::size_t> pm(A.size1());
    lu_factorize(A, pm);
    output_matrix.assign(identity_matrix<double> (A.size1()));
    lu_substitute(A, pm, output_matrix);
    return output_matrix;
}

Kalman::Kalman(mat prediction_matrix, mat observation_model, mat initial_error_covariance, mat process_noise_covariance,
               mat sensor_noise_covariance) :
        prediction_matrix(prediction_matrix), observation_model(observation_model),
        initial_error_covariance(initial_error_covariance), process_noise_covariance(process_noise_covariance),
        sensor_noise_covariance(sensor_noise_covariance) {};

void Kalman::predict(vec& X, mat& P) {
    X = prod(prediction_matrix, X);
    P = prod(mat(prod(prediction_matrix, P)), trans(prediction_matrix)) + process_noise_covariance;
}

void Kalman::correct(vec& X, const vec& Y, mat& P) {
    mat HPH = prod(mat(prod(observation_model, P)), trans(observation_model));
    mat inv = invert_matrix(HPH + sensor_noise_covariance);
    mat K = prod(mat(prod(P, trans(observation_model))), inv);
    X = X + prod(K, vec(Y - prod(observation_model, X)));
    P = prod((identity_matrix<double> (P.size1()) - prod(K, observation_model)), P);
}

void Kalman::execute_step(vec& X, const vec& Y, mat& P) {
    predict(X, P);
    correct(X, Y, P);
}

std::vector<double> Kalman::estimate_position(const std::vector<double>& measured_position,
                                              const std::vector<double>& measured_velocity) {
    vec X(3, 0); // state: position, velocity, velocity_offset
    vec Y(2); // measurements: position, velocity
    mat error_covariance(initial_error_covariance);
    std::vector<double> x_predictions;
    x_predictions.push_back(X(0));
    for (int i = 1; i < measured_position.size(); i++) {
        Y(0) = measured_position[i]; Y(1) = measured_velocity[i];
        execute_step(X, Y, error_covariance);
        x_predictions.push_back(X(0));
    }
    return x_predictions;
}
