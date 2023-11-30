#include "MMVII_SysSurR.h"
#include "MMVII_Sensor.h"

using namespace NS_SymbolicDerivative;

namespace MMVII
{

class cPhotogrammetricProject;
class cSensorCamPC;

/**   Class to represent Topo constraints for bundle adjustment
 */
enum class TopoObsType
{
    dist=3,
    subFrame=11,
    distParam=22,
};


class cBA_Topo
{
public :

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

    void AddEquation_Dist3d(cResolSysNonLinear<tREAL8> &);

    const cPhotogrammetricProject &mPhProj;
    std::vector<double>          mSigma;
    std::vector<double>          mWeight;
    std::map<TopoObsType, cCalculator<double>*> mTopoObsType2equation;
    bool                         mOk;

};


};

