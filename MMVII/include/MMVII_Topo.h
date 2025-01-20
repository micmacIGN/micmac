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
typedef std::map<const cTopoPoint*, std::vector<cTopoObs*>> tSimpleObsMap;

class cBA_Topo : public cMemCheck
{
    friend class cTopoData;
public :
    cBA_Topo(cPhotogrammetricProject *aPhProj);
    ~cBA_Topo();
    void clear();

    void findPtsUnknowns(const cBA_GCP &aBA_GCP, cPhotogrammetricProject *aPhProj); //< to be called after points creation and before AddToSys and ObsSetStation::SetOrigin...

    void  AddToSys(cSetInterUK_MultipeObj<tREAL8> &aSetInterUK); // The system must be aware of all the unknowns

    // fix the variables that are frozen
    void SetFrozenAndSharedVars(cResolSysNonLinear<tREAL8> &)  ;

    //  Do the kernel job : add topo constraints to the system
    void AddTopoEquations(cResolSysNonLinear<tREAL8> &);
    void AddPointsFromDataToGCP(cBA_GCP &aBA_GCP); //< get creates points in gcp from points names in data from mAllTopoDataIn, clear mAllTopoDataIn
    void FromData(const cBA_GCP &aBA_GCP, cPhotogrammetricProject *aPhProj); //< get data from mAllTopoDataIn
    void ToFile(const std::string & aName) const;
    void print();
    void printObs(bool withDetails=false);
    double  Sigma0() {return mSigma0;}
    std::vector<cTopoObs*> GetObsPoint(std::string aPtName) const;

    bool tryInitAll();
    bool tryInit(cTopoPoint & aTopoPt, tStationsMap & stationsMap, tSimpleObsMap &allSimpleObs);

    bool mergeUnknowns(); //< if several stations share origin etc.
    void makeConstraints(cResolSysNonLinear<tREAL8> &aSys);
    const std::map<std::string, cTopoPoint> & getAllPts() const { return mAllPts; }
    std::map<std::string, cTopoPoint> & getAllPts() { return mAllPts; }
    const cTopoPoint & getPoint(std::string name) const;
    cCalculator<double>* getEquation(eTopoObsType tot) const;
    tPtrSysCo getSysCo() const { return mSysCo; }
    const tStationsMap& getAllStations() const { return mAllStations; }
    const tSimpleObsMap& gAllSimpleObs() const { return mAllSimpleObs; }

    friend void BenchTopoComp1example(const std::pair<cTopoData, cSetMesGnd3D>& aBenchData, tREAL4 targetSigma0);
private :
    cTopoData mAllTopoDataIn;
    cPhotogrammetricProject * mPhProj;
    std::map<eTopoObsType, cCalculator<double>*> mTopoObsType2equation;
    std::map<std::string, cTopoPoint> mAllPts;
    std::vector<cTopoObsSet*> mAllObsSets;
    double                       mSigma0;
    bool                        mIsReady; //< if data has been read (via FromFile)
    tPtrSysCo                     mSysCo;

    // maps derived from mAllObsSets to simplify searches.
    tStationsMap mAllStations; //< map of stations from origin names
    tSimpleObsMap mAllSimpleObs; //< map of obs from simple sets, from origin and target names

    // points initialization methods
    bool tryInit3Obs1Station(cTopoPoint & aPtToInit);
    bool tryInitVertStations(cTopoPoint & aPtToInit);
};


};

#endif // TOPO_H
