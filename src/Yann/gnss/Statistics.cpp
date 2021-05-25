#include "Statistics.h"

// ---------------------------------------------------------------
// Moyenne d'un vecteur
// ---------------------------------------------------------------
double Statistics::mean(std::vector<double> X){
    double m = 0;
    for (unsigned i=0; i<X.size(); i++) m += X.at(i);
    return m/X.size();
}

// ---------------------------------------------------------------
// Moyenne L1 d'un vecteur
// ---------------------------------------------------------------
double Statistics::meanAbs(std::vector<double> X){
    double m = 0;
    for (unsigned i=0; i<X.size(); i++) m += std::abs(X.at(i));
    return m/X.size();
}

// ---------------------------------------------------------------
// Erreur MSE d'un vecteur
// ---------------------------------------------------------------
double Statistics::mse(std::vector<double> X){
    double mse = 0;
    for (unsigned i=0; i<X.size(); i++) mse += X.at(i)*X.at(i);
    return mse/X.size();
}

// ---------------------------------------------------------------
// Erreur RMSE d'un vecteur
// ---------------------------------------------------------------
double Statistics::rmse(std::vector<double> X){
    return sqrt(Statistics::mse(X));
}

// ---------------------------------------------------------------
// Ecart-type d'un vecteur
// ---------------------------------------------------------------
double Statistics::sd(std::vector<double> X){
    double bias = Statistics::mean(X);
    double sd = 0;
	for (unsigned i=0; i<X.size(); i++) sd += pow((X.at(i) - bias),2);
	sd /= X.size();
    return sqrt(sd);
}

// ---------------------------------------------------------------
// Minimum d'un vecteur
// ---------------------------------------------------------------
double Statistics::min(std::vector<double> X){
    double min = X.at(0);
    for (unsigned i=0; i<X.size(); i++){
        if (X.at(i) < min) min = X.at(i);
    }
    return min;
}

// ---------------------------------------------------------------
// Maximum d'un vecteur
// ---------------------------------------------------------------
double Statistics::max(std::vector<double> X){
    double max = X.at(0);
    for (unsigned i=0; i<X.size(); i++){
        if (X.at(i) > max) max = X.at(i);
    }
    return max;
}

// ---------------------------------------------------------------
// Argument du minimum d'un vecteur
// ---------------------------------------------------------------
int Statistics::argmin(std::vector<double> X){
    double min = X.at(0); int idx = 0;
    for (unsigned i=0; i<X.size(); i++){
        if (X.at(i) < min) {
            min = X.at(i);
            idx = i;
        }
    }
    return idx;
}

// ---------------------------------------------------------------
// Argument du maximum d'un vecteur
// ---------------------------------------------------------------
int Statistics::argmax(std::vector<double> X){
    double max = X.at(0); int idx = 0;
    for (unsigned i=0; i<X.size(); i++){
        if (X.at(i) > max){
            max = X.at(i);
            idx = i;
        }
    }
    return idx;
}



// ---------------------------------------------------------------
// Moyenne d'un ensemble de coordonnées
// ---------------------------------------------------------------
ECEFCoords Statistics::mean(std::vector<ECEFCoords> pts){
    ECEFCoords mean;
     for (unsigned i=0; i<pts.size(); i++) {
            mean.X += pts.at(i).X;
            mean.Y += pts.at(i).Y;
            mean.Z += pts.at(i).Z;
     }
     mean.scalar(1.0/pts.size());
     return mean;
}

// ---------------------------------------------------------------
// Moyenne L1 d'un ensemble de coordonnées
// ---------------------------------------------------------------
ECEFCoords Statistics::meanAbs(std::vector<ECEFCoords> pts){
    ECEFCoords mean;
     for (unsigned i=0; i<pts.size(); i++) {
            mean.X += std::abs(pts.at(i).X);
            mean.Y += std::abs(pts.at(i).Y);
            mean.Z += std::abs(pts.at(i).Z);
     }
     mean.scalar(1.0/pts.size());
     return mean;
}

// ---------------------------------------------------------------
// Erreur MSE d'un ensemble de coordonnées
// ---------------------------------------------------------------
ECEFCoords Statistics::mse(std::vector<ECEFCoords> pts){
    ECEFCoords mse;
     for (unsigned i=0; i<pts.size(); i++) {
            mse.X += pts.at(i).X*pts.at(i).X;
            mse.Y += pts.at(i).Y*pts.at(i).Y;
            mse.Z += pts.at(i).Z*pts.at(i).Z;
     }
     mse.scalar(1.0/pts.size());
     return mse;
}

// ---------------------------------------------------------------
// Erreur RMSE d'un ensemble de coordonnées
// ---------------------------------------------------------------
ECEFCoords Statistics::rmse(std::vector<ECEFCoords> pts){
    ECEFCoords rmse = Statistics::mse(pts);
    rmse.X = sqrt(rmse.X);
    rmse.Y = sqrt(rmse.Y);
    rmse.Z = sqrt(rmse.Z);
    return rmse;
}

// ---------------------------------------------------------------
// Ecart-type d'un ensemble de coordonnées
// ---------------------------------------------------------------
ECEFCoords Statistics::sd(std::vector<ECEFCoords> pts){
    ECEFCoords bias = Statistics::mean(pts);
    ECEFCoords sd;
	
	for (unsigned i=0; i<pts.size(); i++){
		sd.X += pow((pts.at(i).X - bias.X),2);
		sd.Y += pow((pts.at(i).Y - bias.Y),2);
		sd.Z += pow((pts.at(i).Z - bias.Z),2);
	}
	
	sd.X /= pts.size();
	sd.Y /= pts.size();
	sd.Z /= pts.size();
	
    sd.X = sqrt(sd.X);
    sd.Y = sqrt(sd.Y);
   	sd.Z = sqrt(sd.Z);
	
    return sd;
}

// ---------------------------------------------------------------
// Minimum d'un ensemble de coordonnées
// ---------------------------------------------------------------
ECEFCoords Statistics::min(std::vector<ECEFCoords> pts){
    ECEFCoords min;
    std::vector<double> X, Y, Z;
    for (unsigned i=0; i<pts.size(); i++){
        X.push_back(pts.at(i).X);
        Y.push_back(pts.at(i).Y);
        Z.push_back(pts.at(i).Z);
    }
    min.X = Statistics::min(X);
    min.Y = Statistics::min(Y);
    min.Z = Statistics::min(Z);
    return min;
}

// ---------------------------------------------------------------
// Maximum d'un ensemble de coordonnées
// ---------------------------------------------------------------
ECEFCoords Statistics::max(std::vector<ECEFCoords> pts){
    ECEFCoords max;
    std::vector<double> X, Y, Z;
    for (unsigned i=0; i<pts.size(); i++){
        X.push_back(pts.at(i).X);
        Y.push_back(pts.at(i).Y);
        Z.push_back(pts.at(i).Z);
    }
    max.X = Statistics::max(X);
    max.Y = Statistics::max(Y);
    max.Z = Statistics::max(Z);
    return max;
}

// ---------------------------------------------------------------
// Moyenne d'un ensemble de coordonnées planes
// ---------------------------------------------------------------
ENUCoords Statistics::mean(std::vector<ENUCoords> pts){
    ENUCoords mean;
     for (unsigned i=0; i<pts.size(); i++) {
            mean.E += pts.at(i).E;
            mean.N += pts.at(i).N;
            mean.U += pts.at(i).U;
     }
     mean.E /= pts.size();
     mean.N /= pts.size();
     mean.U /= pts.size();
     return mean;
}

// ---------------------------------------------------------------
// Moyenne L1 d'un ensemble de coordonnées planes
// ---------------------------------------------------------------
ENUCoords Statistics::meanAbs(std::vector<ENUCoords> pts){
    ENUCoords mean;
     for (unsigned i=0; i<pts.size(); i++) {
            mean.E += std::abs(pts.at(i).E);
            mean.N += std::abs(pts.at(i).N);
            mean.U += std::abs(pts.at(i).U);
     }
     mean.E /= pts.size();
     mean.N /= pts.size();
     mean.U /= pts.size();
     return mean;
}

// ---------------------------------------------------------------
// Erreur MSE d'un ensemble de coordonnées planes
// ---------------------------------------------------------------
ENUCoords Statistics::mse(std::vector<ENUCoords> pts){
    ENUCoords mse;
     for (unsigned i=0; i<pts.size(); i++) {
            mse.E += pts.at(i).E*pts.at(i).E;
            mse.N += pts.at(i).N*pts.at(i).N;
            mse.U += pts.at(i).U*pts.at(i).U;
     }
     mse.E /= pts.size();
     mse.N /= pts.size();
     mse.U /= pts.size();
     return mse;
}

// ---------------------------------------------------------------
// Erreur RMSE d'un ensemble de coordonnées planes
// ---------------------------------------------------------------
ENUCoords Statistics::rmse(std::vector<ENUCoords> pts){
    ENUCoords rmse = Statistics::mse(pts);
    rmse.E = sqrt(rmse.E);
    rmse.N = sqrt(rmse.N);
    rmse.U = sqrt(rmse.U);
    return rmse;
}

// ---------------------------------------------------------------
// Ecart-type d'un ensemble de coordonnées planes
// ---------------------------------------------------------------
ENUCoords Statistics::sd(std::vector<ENUCoords> pts){
    ENUCoords bias = Statistics::mean(pts);
    ENUCoords sd;
	
    for (unsigned i=0; i<pts.size(); i++){
		sd.E += pow((pts.at(i).E - bias.E),2);
		sd.N += pow((pts.at(i).N - bias.N),2);
		sd.U += pow((pts.at(i).U - bias.U),2);
	}
	
	sd.E /= pts.size();
	sd.N /= pts.size();
	sd.U /= pts.size();
	
    sd.E = sqrt(sd.E);
    sd.N = sqrt(sd.N);
   	sd.U = sqrt(sd.U);
    return sd;
}

// ---------------------------------------------------------------
// Minimum d'un ensemble de coordonnées planes
// ---------------------------------------------------------------
ENUCoords Statistics::min(std::vector<ENUCoords> pts){
    ENUCoords min;
    std::vector<double> E, N, U;
    for (unsigned i=0; i<pts.size(); i++){
        E.push_back(pts.at(i).E);
        N.push_back(pts.at(i).N);
        U.push_back(pts.at(i).U);
    }
    min.E = Statistics::min(E);
    min.N = Statistics::min(N);
    min.U = Statistics::min(U);
    return min;
}

// ---------------------------------------------------------------
// Maximum d'un ensemble de coordonnées planes
// ---------------------------------------------------------------
ENUCoords Statistics::max(std::vector<ENUCoords> pts){
    ENUCoords max;
    std::vector<double> E, N, U;
    for (unsigned i=0; i<pts.size(); i++){
        E.push_back(pts.at(i).E);
        N.push_back(pts.at(i).N);
        U.push_back(pts.at(i).U);
    }
    max.E = Statistics::max(E);
    max.N = Statistics::max(N);
    max.U = Statistics::max(U);
    return max;
}
