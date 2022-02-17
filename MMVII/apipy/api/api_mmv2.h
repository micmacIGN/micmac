#ifndef API_MMV2_H
#define API_MMV2_H

/**
@file
@brief New methods for python API and existing classes
**/

#include "MMVII_all.h"

namespace MMVII
{

class cAppli_Py : public cMMVII_Appli
{
     public :
        cAppli_Py(const std::vector<std::string> &  ,const cSpecMMVII_Appli &);
        ~cAppli_Py() override;
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
     private :
};

}

//-------------------- new functions ---------------------

//!internal usage
void mmv2_init();


#endif //API_MMV2_H
