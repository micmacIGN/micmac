#include "V1VII.h"

#include "MMVII_util.h"
#include "MMVII_MeasuresIm.h"

namespace MMVII
{

/*   ************************************************* */
/*                                                     */
/*                                                     */
/*   ************************************************* */

#if (MMVII_KEEP_LIBRARY_MMV1)

void ImportMesImV1(std::list<cSetMesPtOf1Im>  & aResult,const std::string & aNameFileMesImV1)
{
    aResult.clear();
    cSetOfMesureAppuisFlottants aMesV1 = StdGetFromPCP(aNameFileMesImV1,SetOfMesureAppuisFlottants);

    for (const auto & aLMes1M : aMesV1.MesureAppuiFlottant1Im())
    {
         cSetMesPtOf1Im aLMesV2(aLMes1M.NameIm());

         for (const auto & aMesV1 : aLMes1M.OneMesureAF1I())
             aLMesV2.AddMeasure(cMesIm1Pt(ToMMVII(aMesV1.PtIm()),aMesV1.NamePt(),1.0));

         aResult.push_back(aLMesV2);
    }
}

cSetMesGnd3D ImportMesGCPV1(const std::string & aNameFileMesGCPV1,const std::string & aNameSet)
{
    cSetMesGnd3D  aResult(aNameSet);
    cDicoAppuisFlottant  aSetMesV1 =   StdGetFromPCP(aNameFileMesGCPV1,DicoAppuisFlottant);

    for (const auto & aMesV1 : aSetMesV1.OneAppuisDAF())
    {
        cMes1Gnd3D  aMesV2(ToMMVII(aMesV1.Pt()),aMesV1.NamePt(),1.0);

	aMesV2.SetSigma2(ToMMVII(aMesV1.Incertitude()));

        aResult.AddMeasure3D(aMesV2);
    }

    return aResult;
}

#else //  MMVII_KEEP_LIBRARY_MMV1
void ImportMesImV1(std::list<cSetMesPtOf1Im>  & aResult,const std::string & aNameFileMesImV1)
{
    MMVII_INTERNAL_ERROR("No ImportMesImV1 ");
}
cSetMesGnd3D ImportMesGCPV1(const std::string & aNameFileMesGCPV1,const std::string & aNameSet)
{
    MMVII_INTERNAL_ERROR("No ImportMesGCPV1");
    return cSetMesGnd3D  (aNameSet);
}
#endif //  MMVII_KEEP_LIBRARY_MMV1



};
