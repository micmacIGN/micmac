#include "ctopopoint.h"
#include "MMVII_PhgrDist.h"
#include "ctopocomp.h"
#include <memory>
namespace MMVII
{

cTopoPoint::cTopoPoint(std::string name, const cPtxd<tREAL8,3>& _coord, bool _isFree) :
    isFree(_isFree), coord(_coord), mName(name)
{
}


void cTopoPoint::PutUknowsInSetInterval() //interval?
{
    mSetInterv->AddOneInterv(coord);
}

void cTopoPoint::OnUpdate()
{
}

std::string cTopoPoint::toString()
{
    std::ostringstream oss;
    oss<<"TopoPoint "<<mName<<" "<<coord;
    return  oss.str();
}

std::vector<int> cTopoPoint::getIndices()
{
    //TODO: compute it in PutUknowsInSetInterval()?
    return {(int)IndOfVal(&coord.x()),(int)IndOfVal(&coord.y()),(int)IndOfVal(&coord.z())};
}


void cTopoPoint::addConstraints(cTopoComp *comp)
{
    if (!isFree)
    {
        /*comp->getSys()->AddEqFixVar(IndOfVal(&coord.x()),coord.x(), 1.0); //if constrained
        comp->getSys()->AddEqFixVar(IndOfVal(&coord.y()),coord.y(), 1.0);
        comp->getSys()->AddEqFixVar(IndOfVal(&coord.z()),coord.z(), 1.0);*/
        comp->getSys()->SetFrozenVar(*this,coord); //if fixed
    }
}

};
