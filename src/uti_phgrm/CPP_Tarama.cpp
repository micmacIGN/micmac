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

#define DEF_OFSET -12349876


void Banniere_Tarama()
{
 std::cout << "*******************************************\n";
 std::cout << "*          T-ableau d'                    *\n";
 std::cout << "*          A-ssemblage pour               *\n";



 // pour Mic               *\n";
 // std::cout << "*          M
}

class cAppliTarama : public cAppliWithSetImage
{
     public :
         cAppliTarama(int argc,char ** argv);

         int mResult;
     private  :
};





cAppliTarama::cAppliTarama(int argc,char ** argv) :
    cAppliWithSetImage(argc-1,argv+1, 0)
    //   !!   TheFlagNoOri , ajoute fin janvier 2017, cree un bug, si cette modif est necessaire
    //  dans certain contexte, contacter MPD pour voir comment gerer tout les cas
{

    NoInit = "XXXXXXXXXX";

    // MemoArg(argc,argv);
    MMD_InitArgcArgv(argc,argv);
    std::string  aDir,aPat,aFullDir;
    std::string Aero;
    int  Zoom = 8;
    std::string  NOREP = "NO-REPERE";
    std::string Repere = NOREP;
    std::string DirOut = "TA";
    double   aZMoy = 0;
    int    aKNadir = -1;
    double aIncidMax = 1e5;
    bool   UnUseAXC = false;



    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(aFullDir,"Full Image (Dir+Pat)", eSAM_IsPatFile)
                    << EAMC(Aero,"Orientation", eSAM_IsExistDirOri),
    LArgMain()
                    << EAM(Zoom,"Zoom",true,"Resolution, (Def=8, must be pow of 2)",eSAM_IsPowerOf2)
                    << EAM(Repere,"Repere",true,"Local coordinate system as created with RepLocBascule",eSAM_IsExistFile)
                    << EAM(DirOut,"Out",true,"Directory for output (Deg=TA)")
                    << EAM(aZMoy,"ZMoy",true,"Average value of Z")
                    << EAM(aKNadir,"KNadir",true,"KBest image or Nadir (when exist)")
                    << EAM(aIncidMax,"IncMax",true,"Maximum incidence of image", eSAM_NoInit)
                    << EAM(UnUseAXC,"UnUseAXC",true,"Internal use for unanamorphosed ortho",eSAM_InternalUse)
    );

    if (!MMVisualMode)
    {
#if (ELISE_windows)
        replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
#endif
        SplitDirAndFile(aDir,aPat,aFullDir);

        StdCorrecNameOrient(Aero,aDir);


        MMD_InitArgcArgv(argc,argv);

        const cInterfChantierNameManipulateur::tSet * aSetIm = mEASF.SetIm();
        bool IsGenBundle = false;
       
        for (int aKIm=0 ; aKIm<int(aSetIm->size()) ; aKIm++)
        {
            const std::string & aNameIm = (*aSetIm)[aKIm];
            cBasicGeomCap3D * aCG =  ICNM()->StdCamGenerikOfNames(Aero,aNameIm);

            if ((aCG!=0) && (aCG->DownCastCS()==0))
                IsGenBundle = true;

        }

        std::string aCom =  MM3dBinFile( "MICMAC" )
                + MMDir() + std::string("include/XML_MicMac/MM-TA.xml ")
                + std::string(" WorkDir=") +aDir +  std::string(" ")
                + std::string(" +PatternAllIm=") + QUOTE(aPat) + std::string(" ")
                + std::string(" +Zoom=") + ToString(Zoom)
                + std::string(" +Aero=") + Aero
                + std::string(" +DirMEC=") + DirOut
                ;

        if (IsGenBundle)
        {
             aCom = aCom + " +UseGenBundle=true  +ModeOriIm=eGeomGen +ZIncIsProp=false ";
        }

        if (EAMIsInit(&aIncidMax))
        {
            aCom = aCom + " +DoIncid=true +IncidMax=" + ToString(aIncidMax) + " " + " +ZMoy=" + ToString(AltiMoy()) + " " ;
;
        }

        if (EAMIsInit(&aKNadir))
            aCom = aCom + " +KBestMasqNadir=" + ToString(aKNadir);

        if (EAMIsInit(&aZMoy))
        {
            aCom = aCom + " +FileZMoy=File-ZMoy.xml"
                    + " +ZMoy=" + ToString(aZMoy);
        }
        if (EAMIsInit(&UnUseAXC)) aCom = aCom + " +UnUseAXC=" + ToString(UnUseAXC);

        if (Repere!=NOREP)
        {
            bool IsOrthoXCste;
            bool IsAnamXsteOfCart;
            if (RepereIsAnam(aDir+Repere,IsOrthoXCste,IsAnamXsteOfCart))
            {
                aCom =    aCom
                        +  std::string(" +DoAnam=true ")
                        +  std::string(" +DoIncid=true ")
                        +  std::string(" +ParamAnam=") + Repere;
            }
            else
            {
                aCom =     aCom    + std::string(" +Repere=") + Repere ;
            }
        }

#if (ELISE_windows)
		aCom = "\"" + aCom + "\"";
#endif		
        std::cout << "Com = " << aCom << "\n";
        mResult = system_call(aCom.c_str());

    }
    else
    {
       mResult = EXIT_SUCCESS;
    }
}

int Tarama_main(int argc,char ** argv)
{
   cAppliTarama anAppli(argc,argv);

   return  anAppli.mResult;
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
