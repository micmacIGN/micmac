#ifndef CTOPOCOMP_H
#define CTOPOCOMP_H

#include "ctopoobsset.h"
#include "cMMVII_Appli.h"
#include "SymbDer/SymbDer_Common.h"

using namespace NS_SymbolicDerivative;

namespace MMVII
{
class cTopoObsSet;
class cTopoComp;

/**
 * @brief The cTopoComp class represents a topometric compensation
 */
class cTopoComp
{
public:
    cTopoComp();
    ~cTopoComp();
    bool OneIteration(); ///< returns true if has to continue iterations
    void print();
    void createEx1();
    cResolSysNonLinear<double>* getSys() const {return mSys;}
    cCalculator<double>* getEquation(TopoObsType tot) const;
    bool verbose {false};
    double getSigma0() const;
private:
    void initializeLeastSquares(); ///< call once after points and obs creation
    bool isInit;
    std::vector<cTopoPoint*> allPts;
    std::vector<std::unique_ptr<cTopoObsSet>> allObsSets;
    cSetInterUK_MultipeObj<double> *mSetIntervMultObj; ///< pointer to be able to delete it before allPts
    cResolSysNonLinear<double>*  mSys;
    std::map<TopoObsType, cCalculator<double>*> mTopoObsType2equation;
};

//-----------------------------------------------------------------------------------
/*
class cAppli_TopoComp : public cMMVII_Appli
{
     public :
        cAppli_TopoComp(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
     private :
        cTopoComp  mTopoComp;
};
*/

};
#endif // CTOPOCOMP_H
