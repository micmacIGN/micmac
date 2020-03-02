#ifndef _cAPPLI_MATCH_TIEP_H_ 
#define _cAPPLI_MATCH_TIEP_H_

namespace MMVII
{

/**   
      This class implement some services common to applications doing tie-points matching
*/
class cBaseMatchTieP : public cMMVII_Appli
{
     public :

        cBaseMatchTieP(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        // int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
        void  PostInit();  ///<  Initialisation not in constructor (need ArgObl/ArgOpt have been executed)
     protected :
        std::string  mNameIm1;                ///< First image
        std::string  mNameIm2;                ///< Second image
        std::vector<cSetAimePCAR> mVSAPc1;    ///<  Aime Carac Point of Im1
        std::vector<cSetAimePCAR> mVSAPc2;    ///<  Aime Carac Point of Im2
};


};

#endif //  _cAPPLI_MATCH_TIEP_H_

