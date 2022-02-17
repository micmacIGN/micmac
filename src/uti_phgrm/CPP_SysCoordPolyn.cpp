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



#define DEF_OFSET -12349876


void Init
     (
         const std::string& aName,
         std::vector<Pt3dr> & aVIn,
         std::vector<Pt3dr> & aVOut
     )
{
    ELISE_fp aFP(aName.c_str(),ELISE_fp::READ);
    string aBuf;
    bool GotEOF = false;
    int aCpt=0;
    while (aFP.fgets(aBuf,GotEOF) && (!GotEOF))
    {
         int aId;
         Pt3dr aPIn,aPOut;
         int aNb = sscanf(aBuf.c_str(),  "%d %lf %lf %lf %lf %lf %lf",
                            &aId,
                            &(aPIn.x),&(aPIn.y),&(aPIn.z),
                            &(aPOut.x),&(aPOut.y),&(aPOut.z)
                          );
         ELISE_ASSERT(aNb==7,"sscanf");
         aVIn.push_back(aPIn);
         aVOut.push_back(aPOut);
         aCpt++;
    }
}

//  Init("../Data/Muru/UTM/Tab-Appr_UTM.txt",aVAprIn,aVAprOut);
//  Init("../Data/Muru/UTM/Tab-Eval_UTM.txt",aVTestIn,aVTestOut);

//  bin/SysCoordPolyn /home/binetr/Data/Muru/UTM/Tab-Appr_UTM.txt mtoto.xml [4,4,1] [0,0,1] Test=/home/binetr/Data/Muru/UTM/Tab-Eval_UTM.txt

int SysCoordPolyn_main(int argc,char ** argv)
{
   std::vector<Pt3dr> aVAprIn,aVAprOut;
   std::vector<Pt3dr> aVTestIn,aVTestOut;
   std::string aNameAppr,aNameTest,aNameXML;

   Pt3di aDegXY,aDegZ;
   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAM(aNameAppr)
                    <<  EAM(aNameXML)
                    << EAM(aDegXY)
                    << EAM(aDegZ),
        LArgMain()  << EAM(aNameTest,"Test",true)
   );

   if(!MMVisualMode)
   {
       if (aNameTest=="")
       {
          aNameTest = aNameAppr;
       }

       // std::string aDirXML = DirOfFile(aNameXML);

       Init(aNameAppr,aVAprIn,aVAprOut);
       Init(aNameTest,aVTestIn,aVTestOut);


       cSysCoord *  aW = cSysCoord::WGS84();
       cSysCoord *  aSC = cSysCoord::ModelePolyNomial
                          (
                                        aDegXY,
                                        aDegXY,
                                        aDegZ,
                                        aW,
                                        aVAprIn,aVAprOut
                          );
       cSystemeCoord  aXmlSC = aSC->ToXML() ;
       MakeFileXML(aXmlSC,aNameXML);
       cSystemeCoord  aXml2  = StdGetObjFromFile<cSystemeCoord>(
                                    aNameXML,
                                    StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                    "SystemeCoord",
                                    "SystemeCoord"
                               );
       cSysCoord *  aSC2 = cSysCoord::FromXML(aXml2,(char *)0);
       Pt3dr aDifMax(0,0,0),aDifMoy(0,0,0);

       double aDMaxPolXML = 0;

       for (int aK=0 ; aK<int(aVTestIn.size()) ; aK++)
       {
             Pt3dr aG = aW->ToGeoC(aVTestIn[aK]);
             Pt3dr aResult = aSC->FromGeoC(aG);
             Pt3dr aDif = (aResult- aVTestOut[aK]).AbsP();

             aDifMoy = aDifMoy + aDif;
             aDifMax = Sup(aDifMax,aDif);
             double aDist = euclid(aResult-aSC2->FromGeoC(aG));
             ElSetMax(aDMaxPolXML,aDist);

             aDist = euclid(aSC->ToGeoC(aResult)-aSC2->ToGeoC(aResult));
             ElSetMax(aDMaxPolXML,aDist);

             //std::cout << aVTestIn[aK] << aResult << " " << aVTestOut[aK] << "\n"; getchar();
       }
       std::cout << "DIF MAX " << aDifMax
                 << " DIF-MOY " << (aDifMoy/aVTestIn.size())
                 << " DMax Polyn-XML " << aDMaxPolXML
                 << "\n";

       return EXIT_SUCCESS;
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
