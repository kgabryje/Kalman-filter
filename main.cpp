#include <vector>
#include <fstream>
#include <boost/numeric/ublas/matrix.hpp>
#include "Kalman.h"

using namespace boost::numeric::ublas;

std::vector<double> read_file(std::string filepath) {
    std::vector<double> file_content;
    std::ifstream input_file(filepath);
    if (input_file) {
        double val;
        while (input_file >> val) {
            file_content.push_back(val);
        }
    }
    input_file.close();
    return file_content;
}

int main() {
    std::vector<double> x_measured = read_file("zmierzone_polozenie.csv");
    std::vector<double> v_measured = read_file("zmierzona_predkosc.csv");
    std::vector<double> x_real = read_file("rzeczywiste_polozenie.csv");
    std::vector<double> x_predictions;

    const double dt = 0.01;

    matrix<double> A(3, 3, 1);
    A(0, 1) = A(0, 2) = dt; // pos(k) = pos(k-1) + dt * v + dt * v_offset
    A(1, 0) = A(1, 2) = A(2, 0) = A(2, 1) = 0;

    matrix<double> P(3, 3, 0);
    P(0, 0) = 0.5;
    P(1, 1) = P(2, 2) = 1.5;

    matrix<double> R(2, 2, 0);
    R(0, 0) = 1000;

    matrix<double> Q(3, 3, 0);
    Q(0, 0) = Q(1, 1) = Q(2, 2) = 0.001;

    matrix<double> H(2, 3, 0); //offset unmeasurable
    H(0, 0) = H(1, 1) = 1;

    Kalman kalman(A, H, P, Q, R);
    x_predictions = kalman.estimate_position(x_measured, v_measured);

    std::ofstream file("x_predictions.csv");
    for (int i = 0; i < x_predictions.size(); i++)
        file << x_predictions[i] << std::endl;
    file.close();

    return 0;
}