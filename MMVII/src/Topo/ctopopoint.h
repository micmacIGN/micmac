#ifndef CTOPOPOINT_H
#define CTOPOPOINT_H

#include "MMVII_SysSurR.h"

namespace MMVII
{
class cTopoComp;

/**
 * @brief The cTopoPoint class represents a 3d point.
 * Its coordinates are the least squares parameters.
 */
class cTopoPoint : public cObjWithUnkowns<tREAL8>
{
public:
    cTopoPoint(std::string name, const cPtxd<tREAL8, 3> &_coord, bool _isFree);
    void PutUknowsInSetInterval() override ;///< describes its unknowns
    void OnUpdate() override;    ///< "reaction" after linear update, eventually update inversion
    std::string toString();
    std::string getName() {return mName;}
    std::vector<int> getIndices();
    void addConstraints(cTopoComp* comp);
    bool isFree;
    cPtxd<tREAL8,3> coord;
protected:
    std::string mName;
};

};
#endif // CTOPOPOINT_H
