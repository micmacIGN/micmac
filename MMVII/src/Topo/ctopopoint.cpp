#include "ctopopoint.h"
#include "MMVII_PhgrDist.h"
#include "ctopocomp.h"
#include <memory>
namespace MMVII
{

cTopoPoint::cTopoPoint(std::string name, const cPtxd<tREAL8,3>& _coord, bool _isFree) :
    cPt3dr_UK(_coord), // TODO: change that
    isFree(_isFree), coord(_coord), mName(name)
{
}


std::string cTopoPoint::toString()
{
    std::ostringstream oss;
    oss<<"TopoPoint "<<mName<<" "<<coord;
    return  oss.str();
}

/*
void cTopoPoint::addConstraints(cTopoData *comp)
{
    if (!isFree)
    {
        comp->getSys()->SetFrozenVarCurVal(*this,coord); //if fixed
    }
}*/

};
