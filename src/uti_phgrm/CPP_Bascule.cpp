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

/*



Bascule ".*.jpg" RadialExtended B2   ImRep=00000002.jpg  P1Rep=[463,1406] P2Rep=[610,284] Teta=90
Bascule ".*.jpg" RadialExtended R2.xml   ImRep=00000002.jpg  P1Rep=[463,1406] P2Rep=[610,284] Teta=45


Bascule ".*.jpg" RadialExtended B2 MesureIm=OutAligned-Im.xml Teta=90
Tarama ".*.jpg" B2

Bascule ".*.jpg" RadialExtended R2.xml MesureIm=OutAligned.xml Teta=180
Tarama ".*.jpg" RadialExtended Repere=R2.xml

Bascule ".*.jpg" RadialExtended R3.xml MesureIm=OutAligned.xml Teta=180

*/

#define DEF_OFSET -12349876

int Bascule_main(int argc,char ** argv)
{
    NoInit = "NoP1P2";
    aNoPt = Pt2dr(123456,-8765432);

    MemoArg(argc,argv);
    std::string  aDir,aPat,aFullDir;
    int ExpTxt=0;


    std::string AeroOut;
    std::string AeroIn;
    std::string PostPlan="_Masq";
    std::vector<std::string> ImPl;
    bool AllPlanExist = false;
    bool UserKeyPlan = false;

    Pt2dr aP1Rep = aNoPt;
    Pt2dr aP2Rep = aNoPt;
    Pt2dr aAxeRep(1,0);
    double aNoVal = 1.234e30;
    double  aTetaRep = aNoVal;
    std::string  aImRep=NoInit;

    std::string  aFileMesureIm = "";

    bool  OrthoCyl = false;

    std::string FileOrthoCyl="EmptyXML.xml";

    double DistFE;
    Pt3dr  Normal;
    Pt3dr  SNormal;
    double aLimBsH;

    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(aFullDir,"Full name (Dir+Pat)" , eSAM_IsPatFile)
                    << EAMC(AeroIn,"Orientation in", eSAM_IsExistDirOri)
                    << EAMC(AeroOut,"Out: orientation or local repair (if postfixed by \"xml\")", eSAM_IsExistFile),
    LArgMain()
                    << EAM(ImPl,"ImPl",true)
                    << EAM(ExpTxt,"ExpTxt",true)
                    << EAM(PostPlan,"PostPlan",true)
                    << EAM(AllPlanExist,"AllPl",true)
                    << EAM(UserKeyPlan,"UserKeyPlan",true)
                    << EAM(aP1Rep,"P1Rep",true)
                    << EAM(aP2Rep,"P2Rep",true)
                    << EAM(aAxeRep,"AxeRep",true)
                    << EAM(aImRep,"ImRep",true)
                    << EAM(aTetaRep,"Teta",true)
                    << EAM(aFileMesureIm,"MesureIm",true)
                    << EAM(OrthoCyl,"OrthoCyl",true,"Generate a locla repair of orthocyl mode")
                    << EAM(DistFE,"DistFS",true,"Distance between to fixe scale, if not given no scaling")
                    << EAM(Normal,"Norm",true,"Target normal for the plane")
                    << EAM(SNormal,"SNorm",true,"\"Symbolic Normal\" (must be X, Y or Z)")
                    << EAM(aLimBsH,"LimBsH",true,"Limit ratio base to high (Def=1e-2)")

    );


    if (aTetaRep != aNoVal)
    {
       aAxeRep = Pt2dr::FromPolar(1.0,aTetaRep * (PI/180.0));
    }

    #if (ELISE_windows)
        replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
    #endif
    SplitDirAndFile(aDir,aPat,aFullDir);

    bool ModeRepere = IsPostfixed(AeroOut) && (StdPostfix(AeroOut) == "xml");

    std::string FileOriInPlan  = "EmptyXML.xml";
    std::string FilePlan="Bascule-Plan.xml";
    std::string FileExport = "Export-Orient-Std.xml";

    std::string FileFixScale  = "EmptyXML.xml";

    std::string aPatternPlan;

    std::string aPrefOut="-";

    std::string aStrRep;
    if (! ImPl.empty())
    {
         aPatternPlan = ".*(";
         aPatternPlan = aPatternPlan + ImPl[0];
         for (int aK=1; aK<int(ImPl.size()) ; aK++)
         {
            aPatternPlan = aPatternPlan + "|" + ImPl[aK];
         }
         aPatternPlan =  aPatternPlan+").*";
    }
    else if (UserKeyPlan)
    {
         aPatternPlan  = "Loc-Key-SetImagePlan";  // Plan Utilisateur
    }
    else
    {
         aPatternPlan  =  "NKS-Set-OfExistingPlan@"+ aPat + "@"+  PostPlan ;  // Plan Utilisateur
    }


    ELISE_ASSERT
    (
            ((aP1Rep !=aNoPt) == (aP2Rep !=aNoPt))
         && ((aP1Rep !=aNoPt) == (aImRep !=NoInit))
       ,"Incoh (Pt) in rep cart"
    );

    bool OrientInPlan = false;

    if (aP1Rep !=aNoPt)
    {
         OrientInPlan = true;
    }

    if (aFileMesureIm!="")
    {
        ELISE_ASSERT
        (
                 IsPostfixed(aFileMesureIm)
             &&( StdPostfix(aFileMesureIm)== "xml"),
             "File Mesure Image has no xml extension"
        );
        aImRep = aFileMesureIm;

       aStrRep = aStrRep + " +FileMesure="+aFileMesureIm;
    }

    if (OrthoCyl)
    {
         ELISE_ASSERT(ModeRepere,"Bad export file for OrthoCyl");
    }

    if (true)
    {
        aStrRep =       aStrRep
                      +  " +X1RC=" + ToString(aP1Rep.x)
                      +  " +Y1RC=" + ToString(aP1Rep.y)
                      +  " +X2RC=" + ToString(aP2Rep.x)
                      +  " +Y2RC=" + ToString(aP2Rep.y)
                      +  " +XAxeRC=" + ToString(aAxeRep.x)
                      +  " +YAxeRC=" + ToString(aAxeRep.y)
                      +  " +ImP1P2="+aImRep
                   ;


// aFileMesureIm
        if (ModeRepere)
        {
             FilePlan = "EmptyXML.xml";
             FileExport  =   "Export-Repere-Cart.xml";
             if (OrthoCyl)
                FileOrthoCyl = "Export-OrthoCyl.xml" ;
             aPrefOut= "";
        }
        else
        {
            if (OrientInPlan)
            {
                FileOriInPlan  = "Bascule-InternePlan.xml";
            }

            if (aFileMesureIm!="")
            {
                FileOriInPlan  = "Bascule-InternePlan-ByFile.xml";
            }
        }
    }

   std::string aStrFixS = "";
   if (EAMIsInit(&DistFE))
   {
       FileFixScale = "Bascule-Fixe-Scale.xml";
       aStrFixS =  " +DistFS=" + ToString(DistFE);
   }

#ifdef ELISE_windows
    std::string aCom =   MMDir() + std::string("bin" ELISE_STR_DIR "Apero ")
                       + MMDir() + std::string("include" ELISE_STR_DIR "XML_MicMac" ELISE_STR_DIR "Apero-Bascule.xml ")
#else
    std::string aCom =   MMDir() + std::string("bin" ELISE_STR_DIR "Apero ")
                       + MMDir() + std::string("include" ELISE_STR_DIR "XML_MicMac" ELISE_STR_DIR "Apero-Bascule.xml ")
#endif
                       + std::string(" DirectoryChantier=") +aDir +  std::string(" ")
                       + std::string(" +PatternAllIm=") + QUOTE(aPat) + std::string(" ")
                       + std::string(" +AeroOut=") + aPrefOut + AeroOut
                       + std::string(" +Ext=") + (ExpTxt?"txt":"dat")
                       + std::string(" +AeroIn=-") + AeroIn
                       + std::string(" +FileBasculePlan=") + FilePlan
                       + std::string(" +FileOriInPlan=") + FileOriInPlan
                       + std::string(" +FileFixScale=") + FileFixScale  + aStrFixS
                       + std::string(" +FileExport=") + FileExport
                       + std::string(" +FileOrthoCyl=") + FileOrthoCyl
                       + std::string(" +PostMasq=") + PostPlan
                       + aStrRep
                    ;


   if (EAMIsInit(&aLimBsH))
       aCom = aCom + std::string(" +LimBsH=") + ToString(aLimBsH);


   if (aPatternPlan!="")
   {
        aCom = aCom + std::string(" +PatternPlan=") + QUOTE(aPatternPlan);
   }



   std::cout << "Com = " << aCom << "\n";
   int aRes = system_call(aCom.c_str());


   return aRes;
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
