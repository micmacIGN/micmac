#define WITH_MMV1_FUNCTION  false

#if (WITH_MMV1_FUNCTION)
#include "V1VII.h"
#endif

#include "MMVII_util.h"
#include "MMVII_MeasuresIm.h"

namespace MMVII
{

/*   ************************************************* */
/*                                                     */
/*                                                     */
/*   ************************************************* */

#if (WITH_MMV1_FUNCTION)

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

cSetMesGCP ImportMesGCPV1(const std::string & aNameFileMesGCPV1,const std::string & aNameSet)
{
    cSetMesGCP  aResult(aNameSet);
    cDicoAppuisFlottant  aSetMesV1 =   StdGetFromPCP(aNameFileMesGCPV1,DicoAppuisFlottant);

    for (const auto & aMesV1 : aSetMesV1.OneAppuisDAF())
    {
        cMes1GCP  aMesV2(ToMMVII(aMesV1.Pt()),aMesV1.NamePt(),1.0);

	aMesV2.SetSigma2(ToMMVII(aMesV1.Incertitude()));

        aResult.AddMeasure(aMesV2);
    }

    return aResult;
}

#else
void ImportMesImV1(std::list<cSetMesPtOf1Im>  & aResult,const std::string & aNameFileMesImV1)
{
    MMVII_INTERNAL_ERROR("No ImportMesImV1 ");
}
cSetMesGCP ImportMesGCPV1(const std::string & aNameFileMesGCPV1,const std::string & aNameSet)
{
    MMVII_INTERNAL_ERROR("No ImportMesGCPV1");
    return cSetMesGCP  (aNameSet);
}
#endif



};
