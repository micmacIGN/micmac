#include "cLSQTemplate.h"

int LSQMatch_Main(int argc,char ** argv)
{
   string aTmpl ="";
   string aImg = "";

   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(aTmpl, "Image Template",  eSAM_IsExistFile)
                     << EAMC(aImg, "Target Image to search for template",  eSAM_IsExistFile),
         LArgMain()   
               );
}
