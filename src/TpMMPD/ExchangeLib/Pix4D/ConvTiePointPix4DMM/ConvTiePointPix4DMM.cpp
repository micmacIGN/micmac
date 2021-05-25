#include "ConvTiePointPix4DMM.h"

std::string BannerConvTiePointPix4DMM()
{
    std::string banniere = "\n";
    banniere += "************************************************************************* \n";
    banniere += "**                                                                     ** \n";
    banniere += "**     Convert tie points from Pix4D format to MicMac format           ** \n";
    banniere += "** Pix4D code tie points in mm (bingo & orima), so need photosite size ** \n";
    banniere += "**                                                                     ** \n";
    banniere += "**     In Pix4d :                                                      ** \n";
    banniere += "**     1) Tie points saved after 1st step called Initial Processing    ** \n";
    banniere += "**     2) Tie points file name : project_name_format.txt               ** \n";
    banniere += "**     3) format : bingo, pix4d, orima                                 ** \n";
    banniere += "**     4) Tie points output folder :                                   ** \n";
    banniere += "**          ...\\project_name\\1_initial\\params\\                     ** \n";
    banniere += "**     5) For bingo & orima : Pix4D exports only filtered tie-points   ** \n";
    banniere += "**         -) Image space spatial grid filter                          ** \n";
    banniere += "**         -) Keep highest multiplicity point in each grid             ** \n";
    banniere += "**     6) For pix4d format : all tie-points are exported               ** \n";
    banniere += "************************************************************************* \n";
    return banniere;
}

int ConvTiePointPix4DMM_main(int argc,char ** argv)
{

    cout<<BannerConvTiePointPix4DMM();
    string aTiePointFile;
    string aOut="_Pix4D";
    double aSzPhotosite = -1.0;
    bool aBin=true;
    bool aToFormatClassic = true;

    ElInitArgMain
    (
          argc, argv,
          LArgMain()
                << EAMC(aTiePointFile, "Pix4D tie point file (format bingo or pix4d)", eSAM_IsExistFile),
          LArgMain()
                << EAM(aSzPhotosite, "PSize",false, "Photosite size (in um) - required if convert from Bingo format")
                << EAM(aOut,"Out",false,"Output Homol folder suffix; Def=_Pix4D")
                << EAM(aBin,"Bin",false,"format homol Micmac (bin(def)/txt)")
                << EAM(aToFormatClassic,"Classic",false,"export to format homol Micmac classic (def=true)")
    );



    ConvTiePointPix4DMM * aConv = new ConvTiePointPix4DMM();
    aConv->ImportTiePointFile(aTiePointFile, aConv->file_type());

    if (aConv->file_type() == _TP_BINGO || aConv->file_type() == _TP_ORIMA)
    {
        if (aSzPhotosite == -1.0)
        {
            cout<<"Please enter photosite size (in um) : "<<endl;
            string aGetSzPhotoSite;
            cin>>aGetSzPhotoSite;
            aSzPhotosite = atof(aGetSzPhotoSite.c_str());
        }
        if (aSzPhotosite <= 0)
        {
            cout<<"Who invents a sensor with photosite size < 0 ?..."<<endl;
            return EXIT_FAILURE;
        }
        aConv->SzPhotosite() = aSzPhotosite;
        if (aConv->file_type() == _TP_BINGO)
            aConv->ReadBingoFile(aTiePointFile);
        if (aConv->file_type() == _TP_ORIMA)
        {
            aConv->SzPhotosite() = aSzPhotosite;
            cout<<"ORIMA not support for now ! "<<endl;
            return EXIT_SUCCESS;
        }

    }
    if (aConv->file_type() == _TP_PIX4D)
    {
        aConv->ReadPix4DFile(aTiePointFile);
    }
    aConv->SuffixOut() = aOut;
    aConv->BinOut() = aBin;
    aConv->exportToMMNewFH();
    if (aToFormatClassic)
        {
            aConv->exportToMMClassicFH("./", aOut, aBin);
        }
 cout<<endl<<endl<<"********  Finish  **********"<<endl;
 return EXIT_SUCCESS;
}
