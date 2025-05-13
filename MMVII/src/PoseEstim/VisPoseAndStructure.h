#ifndef  _MMVII_VISPOSEANDSTRUCTURE_H_
#define  _MMVII_VISPOSEANDSTRUCTURE_H_

#include "cMMVII_Appli.h"
#include "MMVII_Sensor.h"
#include "MMVII_UtiSort.h"
//#include "MMVII_util_tpl.h"
//#include "MMVII_Geom3D.h"

namespace MMVII
{

/* ********************************************************** */
/*                                                            */
/*                     cAppli_VisuPoseStr3D                   */
/*                                                            */
/* ********************************************************** */


class cAppli_VisuPoseStr3D : public cMMVII_Appli
{
public:
    cAppli_VisuPoseStr3D(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

    int Exe() override;
    cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
    cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

private:

    cPhotogrammetricProject   mPhProj;
    std::string               mPatImIn;
    double                    mErrProjMax;
    double                    mCamScale;
    std::string               mOutfile;
    bool                      mBinary;

    void WritePly(cComputeMergeMulTieP * &, const std::vector<cSensorImage *>& );
    double CalculateFDepth(const cPt2di&, const double&);
};

}

#endif // _MMVII_VISPOSEANDSTRUCTURE_H_
