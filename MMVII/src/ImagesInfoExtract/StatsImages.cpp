#include "include/MMVII_all.h"
#include "include/MMVII_2Include_Serial_Tpl.h"


namespace MMVII
{

/**   Application for estimation of discontinuities in images measurement
 */
class cAppli_StatDisc : public cMMVII_Appli
{
     public :
       typedef tREAL8        tElIm;
       typedef cIm2D<tElIm> tIm;

       cAppli_StatDisc(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
         int Exe() override;
	 /*
         cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
         cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
         cAppliBenchAnswer BenchAnswer() const override ; ///< Has it a bench, default : no
         int  ExecuteBench(cParamExeBench &) override ;
	 */

     private :

       cBox2di  mBox1;

};



};
