#include <iostream>
#include <MMVII_PCSens.h>

namespace MMVII {
    void CloseRandom();
}

int main()
{
    MMVII::cMMVII_Appli::InitMMVIIDirs(std::string(getenv("HOME"))
                                       + "/micmac/MMVII/");
    MMVII::InitStandAloneAppli("mimi2007");

    std::string oripath = std::string(getenv("HOME"))
      + "/micmac/MMVII/MMVII-TestDir/Input/Saisies-MMV1/"
      + "MMVII-PhgrProj/Ori/toto/Ori-PerspCentral-IMGP4168.JPG.xml";
    MMVII::cSensorCamPC *aCam;
    aCam = MMVII::cSensorCamPC::FromFile(oripath);
    std::cout<<"Center: "<<aCam->Center()<<".\n";

    delete aCam;
    MMVII::CloseRandom();
    std::cout<<"done."<<std::endl;
    return 0;
}
