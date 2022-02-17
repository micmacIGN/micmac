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

typedef enum
{
  eTS_P,
  eTS_PB,
  eTS_PBR,
  eTS_M,
  eTS_S,
  eTS_NB,
  eTS_NT,
  eTS_MMD,
  eTS_NbVals
} eTypeSEL;

std::string  eToString(const eTypeSEL & anObj)
{
    if (anObj==eTS_P)
       return  "eTS_P";
    if (anObj==eTS_PB)
       return  "eTS_PB";
    if (anObj==eTS_PBR)
       return  "eTS_PBR";
    if (anObj==eTS_M)
       return  "eTS_M";
    if (anObj==eTS_S)
       return  "eTS_S";
    if (anObj==eTS_NB)
       return  "eTS_NB";
    if (anObj==eTS_NT)
       return  "eTS_NT";
    if (anObj==eTS_MMD)
       return  "eTS_MMD";
    if (anObj==eTS_NbVals)
       return  "eTS_NbVals";
  std::cout << "Enum = eTypeSEL\n";
    ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
    return "";
}

void Banniere_SEL()
{
   std::cout << "\n";
   std::cout <<  " *********************************\n";
   std::cout <<  " *     S-aisie d'                *\n";
   std::cout <<  " *     E-lements de              *\n";
   std::cout <<  " *     Liaison                   *\n";
   std::cout <<  " *********************************\n";

}

void Sys(const std::string & aStr)
{
   VoidSystem(aStr.c_str());
}

int SEL_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv);


    Pt2di aSzW(1000,900);
/*
    if (! ELISE_fp::exist_file(MMDir() + "bin/MICMACSaisieLiaisons"))
       VoidSystem("make -f MakeMICMAC  bin/MICMACSaisieLiaisons");
*/

    std::string aDir;
    std::string aN1;
    std::string aN2;
    std::string aKeyH;

    int aRedr=0;
    std::string aFilter="";
    bool aRedrL1 = false;
    bool ModeEpip = false;

    std::string aKeyCompl="Cple2HomAp";

    std::string SH="";
    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAMC(aDir,"Directory", eSAM_IsDir)
                      << EAMC(aN1,"First image name", eSAM_IsExistFile)
                      << EAMC(aN2,"Second image name", eSAM_IsExistFile) ,
           LArgMain() << EAM(aRedr,"R",true)
                      << EAM(aRedrL1,"RL1",true,"Estimate Homography using L1 mode")
                      << EAM(aFilter,"F",true)
                      << EAM(aKeyH,"KH",true,"In P PB PBR M S NB NT MMD",eSAM_None,ListOfVal(eTS_NbVals,"eTS_"))
                      << EAM(aKeyCompl,"KCpl",true)
                      << EAM(aSzW,"SzW",true)
                      << EAM(ModeEpip,"ModeEpip",true,"If mode epip, the y displacement are forced to 0")
                      << EAM(SH,"SH",true,"Homologue extenion for NB/NT mode")
    );

    if (!MMVisualMode)
    {

        std::string aCom =   MM3dBinFile("MICMACSaisieLiaisons")
                           // + MMDir()+std::string("applis/XML-Pattron/Pattron-MicMacLiaison.xml ")
                           + MMDir()+std::string("include/XML_MicMac/Pattron-MicMacLiaison.xml ")
                           + " WorkDir=" + aDir
                           + " %Im1=" + aN1
                           + " %Im2=" + aN2
                           + " %SL_XSzW=" + ToString(aSzW.x)
                           + " %SL_YSzW=" + ToString(aSzW.y)
                           + " %SL_Epip=" + ToString(ModeEpip)
                         ;

        if (aRedr)
           aCom = aCom + " SL_NewRedrCur=true";

        if (aRedrL1)
           aCom = aCom + " SL_L2Estim=false";

       if (aFilter!="")
           aCom = aCom
                  /* + " SL_TJS_FILTER=true" */
              +  " SL_FILTER=" +aFilter;

       if (aKeyH!="")
       {
           if (aKeyH=="P")
           {
              aKeyCompl = "PastisHom";
           }
           else if (aKeyH=="PB")
           {
              aKeyCompl = "Key-Assoc-CpleIm2HomolPastisBin";
           }
           else if (aKeyH=="PBR")
           {
              aKeyCompl = "Key-Assoc-SsRes-CpleIm2HomolPastisBin";
           }
           else if (aKeyH=="M")
           {
              aKeyCompl = "MarcHom";
           }
           else if (aKeyH=="S")
           {
              // aKeyCompl = "Key-Assoc-StdHom";
              aKeyCompl = "NKS-Assoc-CplIm2Hom@-Man@xml";
           }
           else if (aKeyH=="NB")
           {
              aKeyCompl = "NKS-Assoc-CplIm2Hom@"+SH+"@dat";
           }
           else if (aKeyH=="NT")
           {
              aKeyCompl = "NKS-Assoc-CplIm2Hom@"+SH+"@txt";
           }
           else if (aKeyH=="MMD")
           {
              aKeyCompl = "NKS-Assoc-CplIm2Hom@-DenseM@dat";
           }
           else
           {
               std::cout << "For Key=[" << aKeyH << "]\n";
               ELISE_ASSERT(false,"Do Not know key");
           }
       }

       aCom = aCom + " FCND_CalcHomFromI1I2=" + aKeyCompl;


        std::cout << aCom << "\n";
        Sys(aCom);
        Banniere_SEL();

        return 0;
    }
    else
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
