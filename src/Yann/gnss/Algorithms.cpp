#include "Algorithms.h"

#include "Statistics.h"
#include "AtmosphericModel.h"

// -------------------------------------------------------------------------------
// Algorithme de calcul de la vitesse (3D) du récepteur
// -------------------------------------------------------------------------------
Solution Algorithms::estimateSpeed(std::vector<double> doppler, std::vector<ECEFCoords> sat_pos, std::vector<ECEFCoords> sat_speed, ECEFCoords rcv, int freq){

    size_t n = doppler.size();
    double EMISSION_FREQ = L1_FREQ;

    if (freq == 2) EMISSION_FREQ = L2_FREQ;

    // Vecteur d'obs et matrice schéma
	ElMatrix<REAL> A(4,n,0.0);
	ElMatrix<REAL> B(1,n,0.0);
    for (unsigned i=0; i<n; i++){
        ECEFCoords radius = (rcv-sat_pos.at(i)); radius.normalize();
        double radial_speed = radius.dot(sat_speed.at(i));
        A(0,i) = radius.X;
        A(1,i) = radius.Y;
        A(2,i) = radius.Z;
        A(3,i) = -1;
        B(0,i) = -Utils::C*doppler.at(i)/EMISSION_FREQ + radial_speed;    // Signe  !!!!
    }

    // Résolution
    ElMatrix<REAL> X = gaussj(A.transpose()*A)*(A.transpose()*B);

    // Output
    Solution solution;
    ECEFCoords speed(X(0,0), X(0,1), X(0,2));
    solution.setSpeed(speed);
    solution.setClockDrift(X(0,3)/Utils::C);

    return solution;

}


// -------------------------------------------------------------------------------
// Algorithme de calcul de la vitesse (3D) du récepteur
// -------------------------------------------------------------------------------
Solution Algorithms::estimateSpeed(ObservationSlot slot, NavigationData nav, ECEFCoords rcv, int freq){

    std::vector<std::string> sat_names = slot.getSatellites();
    std::vector<double> doppler;
    std::vector<ECEFCoords> sat_pos;
    std::vector<ECEFCoords> sat_speed;

    for (unsigned i=0; i<sat_names.size(); i++){

        // Constellations autorisées pour le calcul
        if ((sat_names.at(i).substr(0,1) == "C")) continue;
        if ((sat_names.at(i).substr(0,1) == "J")) continue;
        if ((sat_names.at(i).substr(0,1) == "S")) continue;
        if ((sat_names.at(i).substr(0,1) == "G") && (!GPS_CONST))      continue;
        if ((sat_names.at(i).substr(0,1) == "R") && (!GLONASS_CONST))  continue;
        if ((sat_names.at(i).substr(0,1) == "E") && (!GALILEO_CONST))  continue;

        // Ephémérides disponibles
        if (!nav.hasEphemeris(sat_names.at(i), slot.getTimestamp())) continue;

        // Test de cohérence du code
        if (slot.getObservation(sat_names.at(i)).getC1() < PSR_ABERRANT) continue;

        // Doppler measurement
        double measure = 0;
        if (freq == 1){
            measure = slot.getObservation(sat_names.at(i)).getD1();
        } else{
            measure = slot.getObservation(sat_names.at(i)).getD2();
        }

        // Test erreur doppler
        if (measure == 0) continue;

        doppler.push_back(measure);

        // Satellite kinematics
        sat_pos.push_back(nav.computeSatellitePos(sat_names.at(i), slot.getTimestamp()));
        sat_speed.push_back(nav.computeSatelliteSpeed(sat_names.at(i), slot.getTimestamp()));

    }

    // Obs suffisantes
    if (doppler.size() < 4){
        Solution solution;
        solution.setCode(1);
        return solution;
    }

    return estimateSpeed(doppler, sat_pos, sat_speed, rcv, freq);

}

// -------------------------------------------------------------------------------
// Algorithme de calcul du vecteur complet PVT du récepteur
// -------------------------------------------------------------------------------
Solution Algorithms::estimateState(ObservationSlot slot, NavigationData nav){

    // Position and time estimation
    Solution solution = estimatePosition(slot, nav);

    // Speed and time drift estimation
    if (!slot.hasObservation("D1")){;
        Solution speed = estimateSpeed(slot, nav, solution.getPosition(), 1);
        solution.setSpeed(speed.getSpeed());
        solution.setClockDrift(speed.getClockDrift());
        return solution;
    }

    if (!slot.hasObservation("D2")){;
        Solution speed = estimateSpeed(slot, nav, solution.getPosition(), 2);
        solution.setSpeed(speed.getSpeed());
        solution.setClockDrift(speed.getClockDrift());
    }

    return solution;

}


// -------------------------------------------------------------------------------
// Algorithme de calcul d'une trajectoire complète
// -------------------------------------------------------------------------------
Trajectory Algorithms::estimateTrajectory(ObservationData obs, NavigationData nav){

    Trajectory trajectory;

    for (unsigned i=0; i<obs.getNumberOfObservationSlots(); i++){

        ObservationSlot slot = obs.getObservationSlots().at(i);
        Solution solution = Algorithms::estimateState(slot, nav);

        if (solution.getCode() != 0) continue;

        trajectory.addPoint(solution);

        std::cout << "Processing GNSS file " << slot.getTimestamp() << " [";
        std::cout << i << "/" << obs.getNumberOfObservationSlots() << "]\r";

    }

    std::cout << std::endl;

    return trajectory;

}


// -------------------------------------------------------------------------------
// Algorithme de calcul d'une position approchée du récepteur sans tenir compte
// des corrections atmosphériques. Prise en compte des pondérations.
// -------------------------------------------------------------------------------
Solution Algorithms::estimateApproxPosition(std::vector<double> psr, std::vector<ECEFCoords> sat_pos, ElMatrix<REAL> SIGMA){

    size_t n = psr.size();
    bool convergence = false;

    ECEFCoords receiver(0,0,0);
    Solution solution;
    solution.setPosition(receiver);

	// Matrice des pondérations
	ElMatrix<REAL> P = gaussj(SIGMA);

    if (n < 4) return solution;

    // Initialisation
    ElMatrix<REAL> X(1,4,0.0);

    // Itérations
    while (!convergence){

        // Design and obs matrices
        ElMatrix<REAL> A(4,n,0.0);
		ElMatrix<REAL> B(1,n,0.0);
        for (unsigned i=0; i<n; i++){

            ECEFCoords sat = sat_pos.at(i);
            double ri = sat.distanceTo(receiver);

            A(0,i) = (X(0,0)-sat.X)/ri;
            A(1,i) = (X(0,1)-sat.Y)/ri;
            A(2,i) = (X(0,2)-sat.Z)/ri;
            A(3,i) = 1.0;

            B(0,i) = psr.at(i) - (ri + X(0,3));

        }

        // Résolution
        ElMatrix<REAL> dX = gaussj(A.transpose()*P*A)*(A.transpose()*P*B);
        X = X + dX;

        receiver.X = X(0,0);
        receiver.Y = X(0,1);
        receiver.Z = X(0,2);

        convergence = (pow(dX(0,0),2) + pow(dX(0,1),2) + pow(dX(0,2),2) < 0.01);

    }

    ECEFCoords position(X(0,0), X(0,1), X(0,2));

    solution.setPosition(position);
	solution.setDeltaTime(X(0,3)/Utils::C);


    return solution;

}

// -------------------------------------------------------------------------------
// Algorithme de calcul des indicateurs DOP d'une configuration sats - rcv
// -------------------------------------------------------------------------------
void Algorithms::computeDopIndices(Solution& solution, std::vector<ECEFCoords> sat_pos){

    double r; ENUCoords temp;
	ECEFCoords rcv = solution.getPosition();

    size_t n = sat_pos.size();

    ElMatrix<REAL> G(4,n,0.0);
    for (unsigned i=0; i<n; i++) {
        r = sat_pos.at(i).distanceTo(rcv);
        temp = sat_pos.at(i).toENUCoords(rcv);
        G(0,i) = -temp.E/r;
        G(1,i) = -temp.N/r;
        G(2,i) = -temp.U/r;
        G(3,i) = 1.0;
    }

    ElMatrix<REAL> DOP = gaussj(G.transpose()*G);

    solution.setVDOP(sqrt(DOP(2, 2)));
    solution.setTDOP(sqrt(DOP(3, 3)));
    solution.setHDOP(sqrt(DOP(0, 0)+DOP(1, 1)));
    solution.setPDOP(sqrt(DOP(0, 0)+DOP(1, 1)+DOP(2, 2)));
    solution.setGDOP(sqrt(DOP(0, 0)+DOP(1, 1)+DOP(2, 2)+DOP(3, 3)));

}

// -------------------------------------------------------------------------------
// Algorithme de calcul des résidus d'une solution GPS
// -------------------------------------------------------------------------------
std::vector<double> Algorithms::computeResiduals(Solution solution, std::vector<std::string> prn, ObservationSlot slot, NavigationData nav){
	std::vector<double> residuals;
	for (unsigned i=0; i<solution.getUsedSatellites().size(); i++){
		std::string prn = solution.getUsedSatellites().at(i);
		double obs = slot.getObservation(prn).getC1();
		double theoric = solution.getPosition().distanceTo(nav.computeSatellitePos(prn, slot.getTimestamp(), obs));
		theoric -= nav.computeSatelliteClockError(prn, slot.getTimestamp())*Utils::C;
		theoric += solution.getDeltaTime()*Utils::C;
		residuals.push_back(obs-theoric);
	}
	return residuals;
}


// -------------------------------------------------------------------------------
// Algorithme de calcul d'une position approchée du récepteur sans tenir compte
// des corrections atmosphériques.
// -------------------------------------------------------------------------------
Solution Algorithms::estimateApproxPosition(std::vector<double> psr, std::vector<ECEFCoords> satellite_pos){

    // Vecteur de poids
    ElMatrix<REAL> W(static_cast<int>(psr.size()), static_cast<int>(psr.size()), 0.0);
    for (unsigned i=0; i<psr.size(); i++) W(i,i) = 1;

    return estimateApproxPosition(psr, satellite_pos, W);

}


// -------------------------------------------------------------------------------
// Algorithme de calcul d'une position finale du récepteur
// -------------------------------------------------------------------------------
Solution Algorithms::estimatePosition(std::vector<std::string> sat_names,
                                      std::vector<double> pseudorange,
                                      std::vector<ECEFCoords> satellite_positions,
                                      AtmosphericModel atm){

    // Solution approchée
    ECEFCoords approx = estimateApproxPosition(pseudorange, satellite_positions).getPosition();

    // Calcul des élévations
    std::vector<double> weights_elevation;
    for (unsigned i=0; i<pseudorange.size(); i++){
        weights_elevation.push_back(approx.elevationTo(satellite_positions.at(i)));
    }

    // Masque d'élévation
    std::vector<double> psr_masked;
    std::vector<double> weights_masked;
    std::vector<ECEFCoords> sat_pos_masked;
    std::vector<std::string> sat_names_masked;
    double threshold = Utils::deg2rad(MASK_ELEV_DEG);
    for (unsigned i=0; i<pseudorange.size(); i++){
        if (weights_elevation.at(i) >= threshold){
            psr_masked.push_back(pseudorange.at(i));
            sat_pos_masked.push_back(satellite_positions.at(i));
            weights_masked.push_back(weights_elevation.at(i));
            sat_names_masked.push_back(sat_names.at(i));
        }
    }

    // Sécurité 1
    if (psr_masked.size() < 4) {
        psr_masked = pseudorange;
        sat_pos_masked = satellite_positions;
        weights_masked = weights_elevation;
        sat_names_masked = sat_names;
    }

    // Corrections atmosphériques
    for (unsigned i=0; i<psr_masked.size(); i++){
        psr_masked.at(i) += atm.all_corrections(approx, sat_pos_masked.at(i));
    }

    // Solution affinée
    Solution solution = estimateApproxPosition(psr_masked, sat_pos_masked);
	ECEFCoords position = solution.getPosition();

    // Analyse des résidus
    std::vector<double> E;
    for (unsigned i=0; i<psr_masked.size(); i++){
        E.push_back(position.distanceTo(sat_pos_masked.at(i)) - psr_masked.at(i) + solution.getDeltaTime()*Utils::C);
    }

    std::vector<double> psr_final;
    std::vector<ECEFCoords> satellite_positions_final;
    std::vector<double> weights_final;
    std::vector<std::string> sat_names_final;

    for (unsigned i=0; i<psr_masked.size(); i++){
        if (std::abs(E[i]) < MASK_PSR_ERROR) {
            psr_final.push_back(psr_masked.at(i));
            satellite_positions_final.push_back(sat_pos_masked.at(i));
            weights_final.push_back(weights_masked.at(i));
            sat_names_final.push_back(sat_names_masked.at(i));
        }
    }

    // Sécurité 2
    if (psr_final.size() < 4) {
        psr_final = psr_masked;
        satellite_positions_final = sat_pos_masked;
        weights_final = weights_masked;
        sat_names_final = sat_names_masked;
    }


	// Elevation mapping function
	std::vector<double> m;
	for (unsigned i=0; i<weights_final.size(); i++){
		m.push_back(0.8/sin(weights_final.at(i)));
	}

	// Calcul des poids de Gauss-Markov
	ElMatrix<REAL> W(static_cast<int>(weights_final.size()), static_cast<int>(weights_final.size()), 0.0);
    for (unsigned i=0; i<weights_final.size(); i++) {
		W(i,i) = WEIGHTED_OLS? m.at(i)*m.at(i):1.0;
	}

    // Solution finale
    Solution output = estimateApproxPosition(psr_final, satellite_positions_final, W);
    output.setNumberOfVisibleSatellites(static_cast<int>(pseudorange.size()));
    output.setUsedSatellites(sat_names_final);

	// ----------------------------------------------
    // Calcul des indicateurs DOP
    // ----------------------------------------------
    computeDopIndices(output, satellite_positions_final);

    return output;

}


// -------------------------------------------------------------------------------
// Algorithme de calcul d'une position finale du récepteur à partir de TAD
// -------------------------------------------------------------------------------
Solution Algorithms::estimatePosition(ObservationSlot slot, NavigationData nav){

    // Recherche des satellites visibles
    std::vector<std::string> sat_names = slot.getSatellites();
    std::vector<std::string> sat_candidates;

    // Formation des obs.
	std::vector<ECEFCoords> sats;
	std::vector<double> psr;

    for (unsigned i=0; i<sat_names.size(); i++){

        // Constellations autorisées pour le calcul
        if ((sat_names.at(i).substr(0,1) == "C")) continue;
        if ((sat_names.at(i).substr(0,1) == "J")) continue;
        if ((sat_names.at(i).substr(0,1) == "S")) continue;
        if ((sat_names.at(i).substr(0,1) == "G") && (!GPS_CONST))      continue;
        if ((sat_names.at(i).substr(0,1) == "R") && (!GLONASS_CONST))  continue;
        if ((sat_names.at(i).substr(0,1) == "E") && (!GALILEO_CONST))  continue;


        // Ephémérides disponibles
        if (!nav.hasEphemeris(sat_names.at(i), slot.getTimestamp())) continue;

        // Mesure de pseudo-distance
        double measure = slot.getObservation(sat_names.at(i)).getC1();

        // Test erreur de code
        if (measure <= PSR_ABERRANT) continue;

        // Correction d'horloge
        NavigationSlot navSlot = nav.getNavigationSlot(sat_names.at(i), slot.getTimestamp());
        double dt = navSlot.computeSatelliteClockError(slot.getTimestamp(), measure);
        measure = measure + dt*Utils::C;

        // Position du satellite
        ECEFCoords pos = navSlot.computeSatellitePos(slot.getTimestamp(), measure);

        sats.push_back(pos);
        psr.push_back(measure);
        sat_candidates.push_back(sat_names.at(i));

    }

    // Pas assez de satellites visibles
    if (psr.size() < 4) {
        Solution solution;
        solution.setCode(1);
        return solution;
    }

    // Corrections atmosphériques
    AtmosphericModel atm(nav);
    atm.setTime(slot.getTimestamp());

    // Calcul de la solution
    Solution solution =  estimatePosition(sat_candidates, psr, sats, atm);;
    solution.setTimestamp(slot.getTimestamp());
    solution.setNumberOfVisibleSatellites(static_cast<int>(sat_names.size()));

	// ----------------------------------------------
    // Calcul des résidus
    // ----------------------------------------------
	solution.setResiduals(computeResiduals(solution, solution.getUsedSatellites(), slot, nav));

    return solution;

}


// -------------------------------------------------------------------------------
// Algorithme de calcul d'une position finale du récepteur à partir du sp3
// -------------------------------------------------------------------------------
Solution Algorithms::estimatePosition(ObservationSlot slot, NavigationData nav, SP3NavigationData sp3_nav){

    // Recherche des satellites visibles
    std::vector<std::string> sat_names = slot.getSatellites();
    std::vector<std::string> sat_candidates;

    // Formation des obs.
	std::vector<ECEFCoords> sats;
	std::vector<double> psr;

    for (unsigned i=0; i<sat_names.size(); i++){

        // Constellations autorisées pour le calcul
        if ((sat_names.at(i).substr(0,1) == "C")) continue;
        if ((sat_names.at(i).substr(0,1) == "J")) continue;
        if ((sat_names.at(i).substr(0,1) == "S")) continue;
        if ((sat_names.at(i).substr(0,1) == "G") && (!GPS_CONST))      continue;
        if ((sat_names.at(i).substr(0,1) == "R") && (!GLONASS_CONST))  continue;
        if ((sat_names.at(i).substr(0,1) == "E") && (!GALILEO_CONST))  continue;

        double measure = slot.getObservation(sat_names.at(i)).getC1();

         // Test erreur de code
        if (measure <= 0) continue;

        // Ephémérides disponibles
        if (!nav.hasEphemeris(sat_names.at(i), slot.getTimestamp())) continue;

        // Correction d'horloge
        double dt = nav.computeSatelliteClockError(sat_names.at(i), slot.getTimestamp(), measure);
        measure = measure + dt*Utils::C;

        // Position du satellite
        ECEFCoords pos = nav.computeSatellitePos(sat_names.at(i), slot.getTimestamp(), measure);

        sats.push_back(pos);
        psr.push_back(measure);
        sat_candidates.push_back(sat_names.at(i));

    }

     // Pas assez de satellites visibles
    if (psr.size() < 4) {
        Solution solution;
        solution.setCode(1);
        return solution;
    }

    // Corrections atmosphériques
    AtmosphericModel atm(nav);
    atm.setTime(slot.getTimestamp());

    // Calcul de la solution
    Solution solution =  estimatePosition(sat_candidates, psr, sats, atm);;
    solution.setTimestamp(slot.getTimestamp());
    solution.setNumberOfVisibleSatellites(static_cast<int>(sat_names.size()));
    return solution;

}


// -------------------------------------------------------------------------------
// Algorithme de calcul d'une position finale du récepteur à partir du sp3
// -------------------------------------------------------------------------------
Solution Algorithms::estimatePosition(ObservationSlot slot, NavigationDataSet nav, AtmosphericModel atm){

    // Recherche des satellites visibles
    std::vector<std::string> sat_names = slot.getSatellites();
    std::vector<std::string> sat_candidate;

    // Formation des obs.
	std::vector<ECEFCoords> sats;
	std::vector<double> psr;

    for (unsigned i=0; i<sat_names.size(); i++){

        // Constellations autorisées pour le calcul
        if ((sat_names.at(i).substr(0,1) == "C")) continue;
        if ((sat_names.at(i).substr(0,1) == "J")) continue;
        if ((sat_names.at(i).substr(0,1) == "S")) continue;
        if ((sat_names.at(i).substr(0,1) == "G") && (!GPS_CONST))      continue;
        if ((sat_names.at(i).substr(0,1) == "R") && (!GLONASS_CONST))  continue;
        if ((sat_names.at(i).substr(0,1) == "E") && (!GALILEO_CONST))  continue;

        double measure = slot.getObservation(sat_names.at(i)).getC1();

         // Test erreur de code
        if (measure <= 0) continue;

        // Ephémérides disponibles
        if (!nav.hasEphemeris(sat_names.at(i), slot.getTimestamp())) continue;


        ECEFCoords pos = nav.computeSatellitePos(sat_names.at(i), slot.getTimestamp(), measure);
        double dt = nav.computeSatelliteClockError(sat_names.at(i), slot.getTimestamp(), measure);

        sats.push_back(pos);
        psr.push_back(measure + dt*Utils::C);
        sat_candidate.push_back(sat_names.at(i));

    }

     // Pas assez de satellites visibles
    if (psr.size() < 4) {
        Solution solution;
        solution.setCode(1);
        return solution;
    }

    // Corrections atmosphériques
    atm.setTime(slot.getTimestamp());

    // Calcul de la solution
    Solution solution =  estimatePosition(sat_candidate, psr, sats, atm);
    solution.setTimestamp(slot.getTimestamp());
    solution.setNumberOfVisibleSatellites(static_cast<int>(sat_names.size()));
    return solution;

}


// -------------------------------------------------------------------------------
// Algorithme de calcul d'une position finale du récepteur en différentiel
// -------------------------------------------------------------------------------
// - rover: slot d'observation de la station à positionner
// - base:  slot d'observation de la station de référence
// - pos:   position ECEF de la station de référence
// - nav:   données de navigation de l'une des deux stations
// -------------------------------------------------------------------------------
// Note : les 2 slots d'observations doivent être de dates proches
// -------------------------------------------------------------------------------
Solution Algorithms::estimateDifferentialPosition(ObservationSlot rover, ObservationSlot base, ECEFCoords pos, NavigationData nav){

     Solution output;

    // Contrôle validité des timestamps
    double time_diff = std::abs(rover.getTimestamp() - base.getTimestamp());

    if (time_diff > TIME_DIFF_TOLERANCE_DGPS){
        std::cout << "Error: time difference (" << time_diff <<" sec) exceeds tolerance for DGPS computation" << std::endl;
        assert(false);
    }

    GPSTime time_rover = rover.getTimestamp();
    GPSTime time_base = base.getTimestamp();

    // Recherche des satellites en commun
    std::vector<std::string> sat_rover = rover.getSatellites();
    std::vector<std::string> sat_base  =  base.getSatellites();

    size_t nb_visible_sats = sat_rover.size();

    for (unsigned i=0; i<sat_rover.size(); i++){
        bool is_in_base = false;
        for (unsigned j=0; j<sat_base.size(); j++){
            if (sat_rover.at(i) == sat_base.at(j)){
                is_in_base = true; break;
            }
        }
        if ((!is_in_base) || (!nav.hasEphemeris(sat_rover.at(i), time_rover))){
            rover.removeSatellite(sat_rover.at(i));
        }
    }

    sat_rover = rover.getSatellites();
    sat_base  =  base.getSatellites();

    for (unsigned i=0; i<sat_base.size(); i++){
        bool is_in_rover = false;
        for (unsigned j=0; j<sat_rover.size(); j++){
            if (sat_base.at(i) == sat_rover.at(j)){
                is_in_rover = true; break;
            }
        }
        if ((!is_in_rover) || (!nav.hasEphemeris(sat_base.at(i), time_base))){
            base.removeSatellite(sat_base.at(i));
        }
    }

    sat_rover = rover.getSatellites();
    sat_base  =  base.getSatellites();

    // Positionnement de la station de base
    Solution solution_base = estimatePosition(base, nav);

    double dt_rcv = solution_base.getDeltaTime();

    // Calcul des corrections sur les pseudo-distances
    std::vector<ECEFCoords> ephemeride;
    std::vector<double> corrected_psr;
    for (unsigned i=0; i<sat_base.size(); i++){

        double psr = base.getObservation(sat_base.at(i)).getC1();
        ECEFCoords sat_pos_base  = nav.computeSatellitePos(sat_base.at(i), time_base,  psr);
        ECEFCoords sat_pos_rover = nav.computeSatellitePos(sat_base.at(i), time_rover, psr);
        ephemeride.push_back(sat_pos_rover);

        double correction = pos.distanceTo(sat_pos_base) - psr + Utils::C*dt_rcv;
        corrected_psr.push_back(rover.getObservation(sat_base.at(i)).getC1() + correction);

    }

     // Calcul des élévations
	ElMatrix<REAL> W(static_cast<int>(sat_base.size()), static_cast<int>(sat_base.size()), 0.0);
    for (unsigned i=0; i<sat_base.size(); i++) {
		W(i,i) = WEIGHTED_OLS? pow(sin(pos.elevationTo(ephemeride.at(i)))/0.8, 2):1.0;
	}

    // Positionnement de la station mobile
    output = estimateApproxPosition(corrected_psr, ephemeride, W);
    output.setNumberOfVisibleSatellites(nb_visible_sats);
    output.setUsedSatellites(sat_base);
    output.setTimestamp(time_rover);

	// ----------------------------------------------
    // Calcul des indicateurs DOP
    // ----------------------------------------------
    computeDopIndices(output, ephemeride);

    return output;

}


// -------------------------------------------------------------------------------
// Algorithme de formation de la matrice des triples différences
// -------------------------------------------------------------------------------
ElMatrix<REAL> Algorithms::makeTripleDifferenceMatrix(int nb_sats, int num_pivot){
	ElMatrix<REAL> TD(4*nb_sats, nb_sats-1, 0.0);
    for (int j=0; j<4; j++){
        int val = 1 - 2*((j==1)||(j==2));
        for (int i=0; i<nb_sats-1; i++){
            int pos = i;
            if (pos >= num_pivot) pos++;
            TD(nb_sats*j+num_pivot,i) = val;
            TD(nb_sats*j+pos,i) = -val;
        }
    }
    return TD;
}


// -------------------------------------------------------------------------------
// Algorithme de calcul d'une position finale du récepteur avec la phase en
// mode différentiel et triple différences. Estimation par filtrage de Kalman.
// -------------------------------------------------------------------------------
// - rover: fichier d'observation de la station mobile
// - base:  fichier d'observation de la station de référence
// - nav:   dataset de navigation (contenant sp3 ou rinex .nav)
// - pos:   positions ECEF de la station de référence
// -------------------------------------------------------------------------------
Solution Algorithms::triple_difference_kalman(ObservationData rover,
                                              ObservationData base,
                                              NavigationDataSet nav,
                                              ECEFCoords ref){


    GPSTime start("21/02/2020 10:23:00");
    GPSTime end("21/02/2020 11:00:00");

    rover.extractSpanTime(start, end);
    base.extractSpanTime(start, end);

    Solution solution;

    double wavelength = Utils::C/L1_FREQ;
    std::string channel = "L1";

    // Matrice identité I3
    ElMatrix<REAL> I3(3,3,0.0);
    I3(0,0) = I3(1,1) = I3(2,2) = 1;

    // Solution initiale
	ElMatrix<REAL> X0(1,3,0.0);

    // Matrice de covariances a priori
	ElMatrix<REAL> Pn(3,3,0.0);
    Pn(0,0) = Pn(1,1) = Pn(2,2) = 1e10;

    // Filtrage de Kalman
    for (unsigned k=1; k<rover.getNumberOfObservationSlots()-1; k++){

        // Récupération des slots d'observation
        ObservationSlot base_slot1  =  base.getObservationSlots().at(k);
        ObservationSlot rover_slot1 = rover.getObservationSlots().at(k);
        ObservationSlot base_slot2  =  base.getObservationSlots().at(k+1);
        ObservationSlot rover_slot2 = rover.getObservationSlots().at(k+1);

        // Timestamps
        GPSTime t1 = base_slot1.getTimestamp();
        GPSTime t2 = base_slot2.getTimestamp();

        std::vector<double> OBS;
        std::vector<std::string> PRN;
        std::vector<ECEFCoords> EPH1;
        std::vector<ECEFCoords> EPH2;

        for (unsigned sr=0; sr<rover_slot1.getNumberOfObservations(); sr++){
            int sb1 = -1;
            int sr2 = -1;
            int sb2 = -1;

            std::string sat_rover = rover_slot1.getSatellites().at(sr);

            // Recherche dans la station de base à l'époque 1
            for (unsigned int i=0; i<base_slot1.getNumberOfObservations(); i++){
                if (sat_rover == base_slot1.getSatellites().at(i)){
                    sb1 = i;
                    break;
                }
            }

            // Recherche dans la station mobile à l'époque 2
            for (unsigned int i=0; i<rover_slot2.getNumberOfObservations(); i++){
                if (sat_rover == rover_slot2.getSatellites().at(i)){
                    sr2 = i;
                    break;
                }
            }

            // Recherche dans la station de base à l'époque 2
            for (unsigned int i=0; i<base_slot2.getNumberOfObservations(); i++){
                if (sat_rover == base_slot2.getSatellites().at(i)){
                    sb2 = i;
                    break;
                }
            }

            if ((sb1 < 0) || (sr2 < 0) || (sb2 < 0)) continue;

            PRN.push_back(sat_rover);
            double psr1 = nav.computeSatellitePos(sat_rover, t1).distanceTo(ref);
            double psr2 = nav.computeSatellitePos(sat_rover, t2).distanceTo(ref);
            ECEFCoords sat_pos_t1 = nav.computeSatellitePos(sat_rover, t1, psr1);
            ECEFCoords sat_pos_t2 = nav.computeSatellitePos(sat_rover, t2, psr2);
            EPH1.push_back(sat_pos_t1);
            EPH2.push_back(sat_pos_t2);

        }

        int N = PRN.size();

        // Matrice de triple différence
        ElMatrix<REAL> TD = makeTripleDifferenceMatrix(N, 0);


        // Matrice de covariance des observations
        ElMatrix<REAL> P = TD*TD.transpose()*1e-10;

        // Observations brutes
         ElMatrix<REAL> PHI(1, PRN.size()*4, 0.0);

        for (int i=0; i<N; i++){
            PHI(0,N*0+i) = rover_slot1.getObservation(PRN.at(i)).getChannel(channel)*wavelength;
            PHI(0,N*1+i) = rover_slot2.getObservation(PRN.at(i)).getChannel(channel)*wavelength;
            PHI(0,N*2+i) = base_slot1.getObservation(PRN.at(i)).getChannel(channel)*wavelength;
            PHI(0,N*3+i) = base_slot2.getObservation(PRN.at(i)).getChannel(channel)*wavelength;
        }

        // TD observées
        ElMatrix<REAL> O = TD*PHI;

        // TD théoriques
        ElMatrix<REAL> G(1, PRN.size()*4, 0.0);


        // Moindres carrés
        for (unsigned iter=0; iter<50; iter++){

            ECEFCoords POS(X0(0,0), X0(0,1), X0(0,2));

            // Observations théoriques des triples différences
            for (int i=0; i<N; i++){
                G(0,N*0+i) = POS.distanceTo(EPH1.at(i));
                G(0,N*1+i) = POS.distanceTo(EPH2.at(i));
                G(0,N*2+i) = ref.distanceTo(EPH1.at(i));
                G(0,N*3+i) = ref.distanceTo(EPH2.at(i));
            }

            ElMatrix<REAL> F = TD*G;

            // Matrice jacobienne en X0
            ElMatrix<REAL> J(3, N-1, 0.0);
            for (int i=1; i<N; i++){

                ECEFCoords EPH11 = EPH1.at(0);
                ECEFCoords EPH12 = EPH2.at(0);
                ECEFCoords EPH21 = EPH1.at(i);
                ECEFCoords EPH22 = EPH2.at(i);

                double Ri1 = POS.distanceTo(EPH11);  double Ri2 = POS.distanceTo(EPH12);
                double Rj1 = POS.distanceTo(EPH21);  double Rj2 = POS.distanceTo(EPH22);

                J(0, i-1) = (POS.X-EPH11.X)/Ri1 - (POS.X-EPH21.X)/Rj1 - (POS.X-EPH12.X)/Ri2 + (POS.X-EPH22.X)/Rj2;
                J(1, i-1) = (POS.Y-EPH11.Y)/Ri1 - (POS.Y-EPH21.Y)/Rj1 - (POS.Y-EPH12.Y)/Ri2 + (POS.Y-EPH22.Y)/Rj2;
                J(2, i-1) = (POS.Z-EPH11.Z)/Ri1 - (POS.Z-EPH21.Z)/Rj1 - (POS.Z-EPH12.Z)/Ri2 + (POS.Z-EPH22.Z)/Rj2;

            }


            ElMatrix<REAL> Y = O-F;
            ElMatrix<REAL> S = J*Pn*J.transpose() + P;
            ElMatrix<REAL> K = Pn*J.transpose()*gaussj(S);

            X0 = X0 + K*Y;
            Pn = (I3-K*J)*Pn;

        }

        ECEFCoords SOL(X0(0,0)-3e-2, X0(0,1), X0(0,2)-3e-2);
        solution.setPosition(SOL);


		//usleep(50000);
        std::cout << "EPOCH " << t1 << "  ";
        std::cout << SOL << "  NSAT = " << N;
        std::cout <<  "   ERR = " << SOL.distanceTo(rover.getApproxAntennaPosition()) << std::endl;

    }

    return solution;

}

// -------------------------------------------------------------------------------
// Algorithme de résolution des ambiüités sur une époque où la position de la
// station mobile (et de la station de base) sont connues exactement.
// -------------------------------------------------------------------------------
double Algorithms::solve_ambiguity_ref(ObservationData rov, ObservationData bas, NavigationData nav, GPSTime time, std::string sat1, std::string sat2){


    double L1 = Utils::C/L1_FREQ;

    ECEFCoords pos_r = rov.getApproxAntennaPosition();
    ECEFCoords pos_b = bas.getApproxAntennaPosition();

    ObservationSlot slot_b = bas.lookForEpoch(time);
    ObservationSlot slot_r = rov.lookForEpoch(time);

    double diff_b = slot_b.getTimestamp()-time;
    double diff_r = slot_r.getTimestamp()-time;

    if (diff_b != 0){
        std::cout << "Warning: base station not synchronized. Time diff = " << diff_b*1e6 << " us" << std::endl;
    }
    if (diff_r != 0){
        std::cout << "Warning: rover station not synchronized. Time diff = " << diff_b*1e6 << " us" << std::endl;
    }

    // Observations
    double C1b = slot_b.getObservation(sat1).getC1(); double L1b = slot_b.getObservation(sat1).getL1();
    double C2b = slot_b.getObservation(sat2).getC1(); double L2b = slot_b.getObservation(sat2).getL1();
    double C1r = slot_r.getObservation(sat1).getC1(); double L1r = slot_r.getObservation(sat1).getL1();
    double C2r = slot_r.getObservation(sat2).getC1(); double L2r = slot_r.getObservation(sat2).getL1();

    // Positions et distances théoriques
    ECEFCoords pos_sat1b = nav.computeSatellitePos(sat1, slot_b.getTimestamp(), C1b); double D1b = pos_b.distanceTo(pos_sat1b);
    ECEFCoords pos_sat2b = nav.computeSatellitePos(sat2, slot_b.getTimestamp(), C2b); double D2b = pos_b.distanceTo(pos_sat2b);
    ECEFCoords pos_sat1r = nav.computeSatellitePos(sat1, slot_r.getTimestamp(), C1r); double D1r = pos_r.distanceTo(pos_sat1r);
    ECEFCoords pos_sat2r = nav.computeSatellitePos(sat2, slot_r.getTimestamp(), C2r); double D2r = pos_r.distanceTo(pos_sat2r);

    // Doubles différences observées et théoriques
    double dd_obs = (L1b - L2b) - (L1r - L2r);
    double dd_thq = ((D1b - D2b) - (D1r - D2r))/L1;

    // Résolution de l'ambigüité entière
    return  dd_obs - dd_thq;

}
