#include <iostream>
#include <cstring>
#include <cstdlib>
#include <stdio.h>
#include <string>
#include <vector>

#include "Gnss.h"

int main(){

    ObservationData obs = RinexReader::readObsFile("C:\\Users\\oo\\Desktop\\komaba_170802_103136.obs"); std::cout << std::endl;
    NavigationData nav  = RinexReader::readNavFile("C:\\Users\\oo\\Desktop\\brdc2140.17n");   std::cout << std::endl;


    for (int i=0; i<obs.getNumberOfObservationSlots(); i++){

        ObservationSlot slot_rover = obs.getObservationSlots().at(i);

        Solution solution = Algorithms::estimateState(slot_rover, nav);

        std::cout << solution.getTimestamp().to_complete_string() << " " << solution.getPosition().toGeoCoords() << " ";
        std::cout << solution.getNumberOfVisibleSatellites() << " " << solution.getNumberOfUsedSatellites() << " ";
        std::cout << " " << solution.getSpeed() << " " << std::endl;

    }



/*
    ObservationData obs = RinexReader::readObsFile("C:\\Users\\oo\\Desktop\\MOBILE.RNX"); std::cout << std::endl;
    NavigationData nav  = RinexReader::readNavFile("C:\\Users\\oo\\Desktop\\brdc0690.17n");   std::cout << std::endl;

    Trajectory trajectory = Algorithms::estimateTrajectory(obs, nav);

    for (int i=1; i<trajectory.getNumberOfPoints(); i++){

        if (trajectory.getPoint(i).getCode() != 0) continue;

        if (trajectory.getPoint(i).getSpeed().norm() < 100/3.6) continue;

        std::cout << trajectory.getPoint(i) << " ";
        std::cout << Utils::formatNumber(trajectory.getPoint(i).getSpeed().norm()*3.6, "%5.2f km/h ");
        std::cout << " " << trajectory.getPoint(i-1).getPosition().distanceTo(trajectory.getPoint(i).getPosition()) << std::endl;

    }


    */
    return 0;

}



