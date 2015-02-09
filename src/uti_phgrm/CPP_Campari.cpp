/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr


    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
#include "StdAfx.h"
#include <algorithm>

/*
Parametre de Tapas :

   - calibration In : en base de donnees ou deja existantes.


*/

// bin/Tapioca MulScale "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" 300 -1 ExpTxt=1
// bin/Tapioca All  "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" -1  ExpTxt=1
// bin/Tapioca Line  "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" -1   3 ExpTxt=1
// bin/Tapioca File  "../micmac_data/ExempleDoc/Boudha/MesCouples.xml" -1  ExpTxt=1

#define DEF_OFSET -12349876

#define  NbModele 10


void Campari_Banniere()
{
    std::cout <<  "\n";
    std::cout <<  " *********************************************\n";
    std::cout <<  " *     C-ompensation of                      *\n";
    std::cout <<  " *     A-lter                                *\n";
    std::cout <<  " *     M-easurements for                     *\n";
    std::cout <<  " *     P-hotomatric                          *\n";
    std::cout <<  " *     A-djustment after                     *\n";
    std::cout <<  " *     R-otation (and position and etc...)   *\n";
    std::cout <<  " *     I-nitialisation                       *\n";
    std::cout <<  " *********************************************\n\n";
}




int Campari_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv);

    std::string aFullDir= "";
    std::string AeroIn= "";
    std::string AeroOut="";

    bool  CPI1 = false;
    bool  CPI2 = false;
    bool  FocFree = false;
    bool  PPFree = false;
    bool  AffineFree = false;
    bool  AllFree = false;

    double aSigmaTieP = 1;
    double aFactResElimTieP = 5;

    std::vector<std::string> GCP;
    std::vector<std::string> EmGPS;
    bool DetailAppuis = false;
    double Viscos = 1.0;
    bool ExpTxt = false;
    std::vector<std::string> aImMinMax;

    Pt3dr aGpsLA;

    int aDegAdd = 0;
    int aDegFree = 0;
    int aDrMax = 0;

    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(aFullDir,"Full Directory (Dir+Pattern)", eSAM_IsPatFile)
                    << EAMC(AeroIn,"Input Orientation", eSAM_IsExistDirOri)
                    << EAMC(AeroOut,"Output Orientation", eSAM_IsOutputDirOri),
    LArgMain()  << EAM(GCP,"GCP",true,"[GrMes.xml,GrUncertainty,ImMes.xml,ImUnc]", eSAM_NoInit)
                    << EAM(EmGPS,"EmGPS",true,"Embedded GPS [Gps-Dir,GpsUnc, ?GpsAlti?], GpsAlti if != Plani", eSAM_NoInit)
                    << EAM(aGpsLA,"GpsLa",true,"Gps Lever Arm, in combination with EmGPS", eSAM_NoInit)
                    << EAM(aSigmaTieP,"SigmaTieP", true, "Sigma use for TieP weighting (Def=1)")
                    << EAM(aFactResElimTieP,"FactElimTieP", true, "Fact elimination of tie point (prop to SigmaTieP, Def=5)")
                    << EAM(CPI1,"CPI1",true,"Calib Per Im, Firt time", eSAM_IsBool)
                    << EAM(CPI2,"CPI2",true,"Calib Per Im, After first time, reUsing Calib Per Im As input", eSAM_IsBool)
                    << EAM(FocFree,"FocFree",true,"Foc Free (Def=false)", eSAM_IsBool)
                    << EAM(PPFree,"PPFree",true,"Principal Point Free (Def=false)", eSAM_IsBool)
                    << EAM(AffineFree,"AffineFree",true,"Affine Parameter (Def=false)", eSAM_IsBool)
                    << EAM(AllFree,"AllFree",true,"Affine Parameter (Def=false)", eSAM_IsBool)
                    << EAM(DetailAppuis,"DetGCP",true,"Detail on GCP (Def=false)", eSAM_IsBool)
                    << EAM(Viscos,"Visc",true,"Viscosity in Levenberg-Marquardt like resolution (Def=1.0)")
                    << EAM(ExpTxt,"ExpTxt",true, "Export in text format (Def=false)",eSAM_IsBool)
                    << EAM(aImMinMax,"ImMinMax",true, "Im max and min to avoid tricky pat")
                    << EAM(aDegAdd,"DegAdd",true, "When specified, degree of additionnal parameter")
                    << EAM(aDegFree,"DegFree",true, "When specified degree of freedom of parameters generiqs")
                    << EAM(aDrMax,"DRMax",true, "When specified degree of freedom of radial parameters")

    );

    if (!MMVisualMode)
    {
        std::string aDir,aPat;
    #if (ELISE_windows)
         replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
    #endif
        SplitDirAndFile(aDir,aPat,aFullDir);
        StdCorrecNameOrient(AeroIn,aDir);

        Pt2dr Focales(0,100000);
        std::string aParamPatFocSetIm = "@" + aPat + "@" + ToString(Focales.x) + "@" + ToString(Focales.y) ;
        std::string aSetIm = "NKS-Set-OfPatternAndFoc" + aParamPatFocSetIm;


        if (EAMIsInit(&aImMinMax))
        {
            ELISE_ASSERT(aImMinMax.size()==2,"Bad size in vect");
            aSetIm =  "NKS-Set-OfPatternAndFocAndInterv" + aParamPatFocSetIm + "@" + aImMinMax[0] + "@" + aImMinMax[1];
        }




       std::string aCom =     MM3dBinFile_quotes( "Apero" )
                           +  ToStrBlkCorr( Basic_XML_MM_File("Apero-Compense.xml") )
                           +  std::string(" DirectoryChantier=") + aDir + " "
                           +  std::string(" +SetIm=") + QUOTE(aSetIm) + " "
                           +  std::string(" +AeroIn=-") + AeroIn + " "
                           +  std::string(" +AeroOut=-") + AeroOut + " "
                          ;

        if (CPI1 || CPI2) aCom       += " +CPI=true ";
        if (CPI2) aCom       += " +CPIInput=true ";
        if (FocFree) aCom    += " +FocFree=true ";
        if (PPFree) aCom    += " +PPFree=true ";
        if (AffineFree) aCom += " +AffineFree=true ";
        if (AllFree) aCom    += " +AllFree=true ";
        if (ExpTxt) aCom += std::string(" +Ext=") + (ExpTxt?"txt ":"dat ")  ;

        if (EAMIsInit(&aFactResElimTieP))
           aCom =  aCom+ " +FactMaxRes=" + ToString(aFactResElimTieP);


       if (EAMIsInit(&Viscos)) aCom  +=  " +Viscos=" + ToString(Viscos) + " ";

       if (EAMIsInit(&DetailAppuis)) aCom += " +DetailAppuis=" + ToString(DetailAppuis) + " ";

        if (EAMIsInit(&GCP))
        {
            ELISE_ASSERT(GCP.size()==4,"Mandatory part of GCP requires 4 arguments");
            double aGcpGrU = RequireFromString<double>(GCP[1],"GCP-Ground uncertainty");
            double aGcpImU = RequireFromString<double>(GCP[3],"GCP-Image  uncertainty");

            std::cout << "THAT IS ::: " << aGcpGrU << " === " << aGcpImU << "\n";

            aCom =   aCom
                   + std::string("+WithGCP=true ")
                   + std::string("+FileGCP-Gr=") + GCP[0] + " "
                   + std::string("+FileGCP-Im=") + GCP[2] + " "
                   + std::string("+GrIncGr=") + ToString(aGcpGrU) + " "
                   + std::string("+GrIncIm=") + ToString(aGcpImU) + " ";
        }
        if (aDegAdd>0)  aCom = aCom + " +HasModeleAdd=true  +ModeleAdditionnel=eModelePolyDeg" +  ToString(aDegAdd);
        if (aDegFree>0)  aCom = aCom + " +DegGen=" +  ToString(aDegFree);
        if (aDrMax>0)   aCom = aCom + " +DRMax=" +  ToString(aDrMax);

        if (EAMIsInit(&EmGPS))
        {
            ELISE_ASSERT((EmGPS.size()>=2) && (EmGPS.size()<=3) ,"Mandatory part of EmGPS requires 2 arguments");
            StdCorrecNameOrient(EmGPS[0],aDir);
            double aGpsU = RequireFromString<double>(EmGPS[1],"GCP-Ground uncertainty");
            double aGpsAlti = aGpsU;
            if (EmGPS.size()>=3)
               aGpsAlti = RequireFromString<double>(EmGPS[2],"GCP-Ground Alti uncertainty");
            aCom = aCom +  " +BDDC=" + EmGPS[0]
                        +  " +SigmGPS=" + ToString(aGpsU)
                        +  " +SigmGPSAlti=" + ToString(aGpsAlti)
                        +  " +WithCenter=true";

            if (EAMIsInit(&aGpsLA))
            {
                aCom = aCom + " +WithLA=true +LaX="  + ToString(aGpsLA.x)
                                         + " +LaY=" + ToString(aGpsLA.y)
                                         + " +LaZ=" + ToString(aGpsLA.z)
                                         + " ";
            }
        }

        if (EAMIsInit(&aSigmaTieP)) aCom = aCom + " +SigmaTieP=" + ToString(aSigmaTieP);


        std::cout << aCom << "\n";
        int aRes = System(aCom.c_str());

        Campari_Banniere();
        BanniereMM3D();

        return aRes;
    }
    else
        return EXIT_SUCCESS;
}


int AperoProg_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv);

    std::string aFullDir= "";
    std::string AeroIn= "";
    std::string AeroOut="";


    /*double aSigmaTieP = 1;
    double aFactResElimTieP = 5;
    double Viscos = 1.0;
    bool ExpTxt = false;*/

    ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(aFullDir,"Full Directory (Dir+Pattern)", eSAM_IsPatFile)
                     << EAMC(AeroIn,"Input Orientation", eSAM_IsExistDirOri)
                     << EAMC(AeroOut,"Output Orientation", eSAM_IsOutputDirOri),
         LArgMain()
    );
    if (!MMVisualMode)
    {
        std::string aDir,aPat;
    #if (ELISE_windows)
         replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
    #endif
        SplitDirAndFile(aDir,aPat,aFullDir);
        StdCorrecNameOrient(AeroIn,aDir);
    }

    return EXIT_SUCCESS;
}



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
