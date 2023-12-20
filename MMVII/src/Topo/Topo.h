#include "MMVII_SysSurR.h"
#include "MMVII_Sensor.h"
#include "ctopodata.h"


using namespace NS_SymbolicDerivative;

namespace MMVII
{

class cPhotogrammetricProject;
class cSensorCamPC;
class cPt3dr_UK;

class cBA_Topo
{
public :
    typedef std::pair<cObjWithUnkowns<tREAL8>*, cPt3dr*> tTopoPtUK;

    cBA_Topo(const cPhotogrammetricProject &, const std::string &aTopoFilePath);
    ~cBA_Topo();

    // The system must be aware of all the unknowns
    void  AddToSys(cSetInterUK_MultipeObj<tREAL8> &);

    // fix the variable that are frozen
    void SetFrozenVar(cResolSysNonLinear<tREAL8> &)  ;

    //  Do the kernel job : add topo constraints to the system
    void AddTopoEquations(cResolSysNonLinear<tREAL8> &);

    void Save();
    bool isOk() const {return mOk;}
private :

    double AddEquation_Dist3d(cResolSysNonLinear<tREAL8> &);

    const cPhotogrammetricProject &mPhProj;

    std::map<eTopoObsType, cCalculator<double>*> mTopoObsType2equation;
    cTopoData                    mTopoData;
    bool                         mOk;
    std::string                  mInFile;

    // tmp: obs here, TODO: use cTopoObsSet
    std::vector<tTopoPtUK>      mPts_UK;

};


};

