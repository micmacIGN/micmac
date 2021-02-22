#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include "Utils.h"
#include "Solution.h"
#include "Trajectory.h"
#include "ECEFCoords.h"
#include "NavigationData.h"
#include "NavigationDataSet.h"
#include "SP3NavigationData.h"
#include "ObservationSlot.h"
#include "AtmosphericModel.h"


class Algorithms{

	private:

        static void computeDopIndices(Solution&, std::vector<ECEFCoords>);
		static std::vector<double> computeResiduals(Solution, std::vector<std::string>, ObservationSlot, NavigationData);
	
    public:

        // Calcul approché
        static Solution estimateApproxPosition(std::vector<double>, std::vector<ECEFCoords>);
        static Solution estimateApproxPosition(std::vector<double>, std::vector<ECEFCoords>, ElMatrix<REAL>);

        // Calcul complet de la position avec sélection des satellitess
        static Solution estimatePosition(std::vector<std::string>, std::vector<double>, std::vector<ECEFCoords>, AtmosphericModel);

        // Calcul de la position (single et DGPS) par TAD
        static Solution estimatePosition(ObservationSlot, NavigationData);
        static Solution estimatePosition(ObservationSlot, NavigationData, SP3NavigationData);
        static Solution estimatePosition(ObservationSlot, NavigationDataSet, AtmosphericModel);
        static Solution estimateDifferentialPosition(ObservationSlot, ObservationSlot, ECEFCoords, NavigationData);

        // Calcul de la vitesse
        static Solution estimateSpeed(std::vector<double>, std::vector<ECEFCoords>, std::vector<ECEFCoords>, ECEFCoords, int);
        static Solution estimateSpeed(ObservationSlot, NavigationData, ECEFCoords, int);

        // Calcul position + vitesse
        static Solution estimateState(ObservationSlot, NavigationData);

        // Calcul d'un fichier complet
        static Trajectory estimateTrajectory(ObservationData, NavigationData);
	
		// Calcul de la position par la phase
        static Solution triple_difference_kalman(ObservationData, ObservationData, NavigationDataSet, ECEFCoords);
		static ElMatrix<REAL> makeTripleDifferenceMatrix(int, int);

};

#endif
