#ifndef TOPO_H
#define TOPO_H

#include "MMVII_SysSurR.h"
#include "MMVII_Sensor.h"
#include "MMVII_SysCo.h"
#include "../src/Topo/ctopopoint.h"
#include "../src/Topo/ctopoobs.h"
#include "../src/Topo/ctopoobsset.h"
#include "../src/Topo/ctopodata.h"

using namespace NS_SymbolicDerivative;

namespace MMVII
{

class cMMVII_BundleAdj;
class cPhotogrammetricProject;
class cSensorCamPC;
class cBA_GCP;

typedef std::map<const cTopoPoint*, std::vector<cTopoObsSetStation*>> tStationsMap;

class cBA_Topo : public cMemCheck
{
    friend class cTopoData;
public :
    cBA_Topo(cPhotogrammetricProject *aPhProj, cMMVII_BundleAdj *aBA);
    ~cBA_Topo();
    void clear();

    void findPtsUnknowns(const std::vector<cBA_GCP *> &vGCP, cPhotogrammetricProject *aPhProj); //< to be called after points creation and before AddToSys and ObsSetStation::SetOrigin...

    void  AddToSys(cSetInterUK_MultipeObj<tREAL8> &); // The system must be aware of all the unknowns

    // fix the variables that are frozen
    void SetFrozenAndSharedVars(cResolSysNonLinear<tREAL8> &)  ;

    //  Do the kernel job : add topo constraints to the system
    void AddTopoEquations(cResolSysNonLinear<tREAL8> &);
    void AddPointsFromDataToGCP(cSetMesImGCP &aFullMesGCP, std::vector<cBA_GCP*> * aVGCP); //< get creates points in gcp from points names in data from mAllTopoDataIn, clear mAllTopoDataIn
    void FromData(const std::vector<cBA_GCP *> &vGCP, cPhotogrammetricProject *aPhProj); //< get data from mAllTopoDataIn
    void ToFile(const std::string & aName) const;
    void print();
    void printObs(bool withDetails=false);
    double  Sigma0() {return mSigma0;}
    std::vector<cTopoObs*> GetObsPoint(std::string aPtName) const;

    bool tryInitAll();
    bool tryInit(cTopoPoint & aTopoPt, tStationsMap & stationsMap);

    bool mergeUnknowns(cResolSysNonLinear<tREAL8> &aSys); //< if several stations share origin etc.
    void makeConstraints(cResolSysNonLinear<tREAL8> &aSys);
    const std::map<std::string, cTopoPoint> & getAllPts() const { return mAllPts; }
    std::map<std::string, cTopoPoint> & getAllPts() { return mAllPts; }
    const cTopoPoint & getPoint(std::string name) const;
    cCalculator<double>* getEquation(eTopoObsType tot) const;
    tPtrSysCo getSysCo() const { return mSysCo; }

    friend void BenchTopoComp1example(const std::pair<cTopoData, cSetMesGCP>& aBenchData, tREAL4 targetSigma0);
private :
    cTopoData mAllTopoDataIn;
    cPhotogrammetricProject * mPhProj;
    std::map<eTopoObsType, cCalculator<double>*> mTopoObsType2equation;
    std::map<std::string, cTopoPoint> mAllPts;
    std::vector<cTopoObsSet*> mAllObsSets;
    double                       mSigma0;
    bool                        mIsReady; //< if data has been read (via FromFile)
    tPtrSysCo                     mSysCo;
};


};

#endif // TOPO_H
