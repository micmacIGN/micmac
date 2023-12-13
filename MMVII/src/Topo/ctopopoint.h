#ifndef CTOPOPOINT_H
#define CTOPOPOINT_H

#include "../BundleAdjustment/BundleAdjustment.h"

namespace MMVII
{
/**
 * @brief The cTopoPoint class represents a 3d point.
 * Its coordinates are the least squares parameters.
 */
class cTopoPoint: public cPt3dr_UK
{
public:
    cTopoPoint(std::string name, const cPtxd<tREAL8, 3> &_coord, bool _isFree);
    std::string toString();
    std::string getName() {return mName;}
    std::vector<int> getIndices();
    //void addConstraints(cTopoComp* comp);
    bool isFree;
    cPtxd<tREAL8,3> coord;
protected:
    std::string mName;
};

};
#endif // CTOPOPOINT_H
