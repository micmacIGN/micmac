#ifndef CTOPODATA_H
#define CTOPODATA_H

#include "ctopoobsset.h"
#include "cMMVII_Appli.h"
#include "SymbDer/SymbDer_Common.h"

using namespace NS_SymbolicDerivative;

namespace MMVII
{
class cTopoObsSet;
class cTopoComp;

/**
 * @brief The cTopoData class represents topometric data
 */
class cTopoData
{
public:
    cTopoData();
    ~cTopoData();
    void print();
    void createEx1();
    cCalculator<double>* getEquation(TopoObsType tot) const;
private:
    std::vector<cTopoPoint*> allPts;
    std::vector<std::unique_ptr<cTopoObsSet>> allObsSets;
    cSetInterUK_MultipeObj<double> *mSetIntervMultObj; ///< pointer to be able to delete it before allPts
    cResolSysNonLinear<double>*  mSys;
    std::map<TopoObsType, cCalculator<double>*> mTopoObsType2equation;
};


};
#endif // CTOPODATA_H
