#include "ctopopoint.h"
#include "MMVII_PhgrDist.h"
#include <memory>
namespace MMVII
{

cTopoPoint::cTopoPoint(std::string name, const cPtxd<tREAL8,3>& _coord, bool _isFree) :
    cPt3dr_UK(_coord), // TODO: change that
    isFree(_isFree), mName(name)
{
}

void cTopoPoint::AddData(const  cAuxAr2007 & anAuxInit)
{
     cAuxAr2007 anAux("TopoPoint",anAuxInit);

     MMVII::AddData(cAuxAr2007("Name",anAux),mName);
     MMVII::AddData(cAuxAr2007("isFree",anAux),isFree);
}

void AddData(const cAuxAr2007 & anAux, cTopoPoint & aTopoPoint)
{
     aTopoPoint.AddData(anAux);
}

void AddData(const cAuxAr2007 & anAux, cTopoPoint * aTopoPoint)
{
     aTopoPoint->AddData(anAux);
}


std::string cTopoPoint::toString()
{
    std::ostringstream oss;
    oss<<"TopoPoint "<<mName<<" "<<Pt();
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
