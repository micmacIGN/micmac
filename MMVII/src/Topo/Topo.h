#include "MMVII_SysSurR.h"
#include "MMVII_Sensor.h"
#include "ctopodata.h"


using namespace NS_SymbolicDerivative;

namespace MMVII
{

class cMMVII_BundleAdj;
class cPhotogrammetricProject;
class cSensorCamPC;
class cPt3dr_UK;

typedef std::pair<cObjWithUnkowns<tREAL8>*, cPt3dr*> tTopoPtUK;

class cBA_Topo
{
public :

    cBA_Topo(const cPhotogrammetricProject &aPhProj, const std::string &aTopoFilePath);
    ~cBA_Topo();

    // The system must be aware of all the unknowns
    void  AddToSys(cSetInterUK_MultipeObj<tREAL8> &);

    // fix the variable that are frozen
    void SetFrozenVar(cResolSysNonLinear<tREAL8> &)  ;

    //  Do the kernel job : add topo constraints to the system
    void AddTopoEquations(cResolSysNonLinear<tREAL8> &);

    void Save();
    void addPointWithUK(const std::string &aName, cObjWithUnkowns<tREAL8>* aUK, cPt3dr* aPt);
    tTopoPtUK& getPointWithUK(const std::string &aName); // fill mPts_UK map
    cCalculator<double>* getEquation(eTopoObsType tot) const;
private :

    //double AddEquation_Dist3d(cResolSysNonLinear<tREAL8> &);

    const cPhotogrammetricProject &mPhProj;

    std::map<eTopoObsType, cCalculator<double>*> mTopoObsType2equation;
    cTopoData                    mTopoData;
    std::string                  mInFile;

    // tmp: obs here, TODO: use cTopoObsSet
    std::map<std::string, tTopoPtUK>      mPts_UK;

};


};

