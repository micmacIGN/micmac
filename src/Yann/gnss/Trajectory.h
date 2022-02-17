#ifndef TRAJECTORY_H
#define TRAJECTORY_H

#include <string>
#include <vector>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <stdio.h>
#include <cmath>

#include "Solution.h"

class Trajectory{

    private:

        std::vector<Solution> points;


    public:

        std::vector<Solution>& getPoints(){return this->points;}
        Solution& getPoint(int index){return this->points.at(index);}
        size_t getNumberOfPoints(){return this->points.size();}

        void setPoints(std::vector<Solution> points){this->points = points;}
        void addPoint(Solution point){this->points.push_back(point);}




};

#endif // TRAJECTORY_H
