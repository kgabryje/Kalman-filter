#include "Kalman.h"

using namespace boost::numeric::ublas;

mat invert_matrix(const mat& input_matrix) {
    mat A(input_matrix);
    mat output_matrix(A.size1(), A.size1());
    permutation_matrix<std::size_t> pm(A.size1());
    lu_factorize(A, pm);
    output_matrix.assign(identity_matrix<double> (A.size1()));
    lu_substitute(A, pm, output_matrix);
    return output_matrix;
}

Kalman::Kalman(mat A, mat H, mat P, mat Q, mat R) : prediction_matrix(A),  observation_model(H), error_covariance(P),
                                                    process_noise_covariance(Q), sensor_noise_covariance(R) {};

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

void Kalman::execute_step(vec& X, const vec& Y) {
    predict(X, error_covariance);
    correct(X, Y, error_covariance);
}

std::vector<double> Kalman::estimate_position(const std::vector<double>& measured_position,
                                              const std::vector<double>& measured_velocity) {
    vec X(3, 0); // state: position, velocity, velocity_offset
    vec Y(2); // measurements: position, velocity
    std::vector<double> x_predictions;
    x_predictions.push_back(X(0));
    for (int i = 1; i < measured_position.size(); i++) {
        Y(0) = measured_position[i]; Y(1) = measured_velocity[i];
        execute_step(X, Y);
        x_predictions.push_back(X(0));
    }
    return x_predictions;
}
