#ifndef CTOPODATA_H
#define CTOPODATA_H

#include "ctopoobsset.h"
//#include "cMMVII_Appli.h"
//#include "SymbDer/SymbDer_Common.h"

using namespace NS_SymbolicDerivative;

namespace MMVII
{
class cTopoObsSet;
class cBA_Topo;

/**
 * @brief The cTopoData class represents topometric data
 */
class cTopoData : public cMemCheck
{
    friend class cBA_Topo;
public:
    cTopoData(const std::string &aName, cBA_Topo* aBA_Topo);
    ~cTopoData();
    void AddData(const  cAuxAr2007 & anAuxInit);
    void ToFile(const std::string & aName) const;
    //void FromFile(const std::string & aName);
    bool FromCompFile(const std::string & aName);
    void print();
    void createEx1();
    void createEx2();
    //cCalculator<double>* getEquation(eTopoObsType tot) const;
private:
    cTopoData(cTopoData const&) = delete;
    cTopoData& operator=(cTopoData const&) = delete;
    bool addObs(int code, const std::string & nameFrom, const std::string & nameTo, double val, double sigma);

    cBA_Topo* mBA_Topo;
    //std::vector<cTopoPoint*> allPts;
    std::vector<std::unique_ptr<cTopoObsSet>> allObsSets;
    cSetInterUK_MultipeObj<double> *mSetIntervMultObj; ///< pointer to be able to delete it before allPts
    //cResolSysNonLinear<double>*  mSys;
    //std::map<eTopoObsType, cCalculator<double>*> mTopoObsType2equation;
};


///  Global function with standard interface required for serialization => just call member
void AddData(const cAuxAr2007 & anAux, cTopoData & aTopoData) ;


};
#endif // CTOPODATA_H
