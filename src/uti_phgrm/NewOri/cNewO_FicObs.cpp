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

#include "NewOri.h"


class cAppliFictObs : public cCommonMartiniAppli
{
    public:
        cAppliFictObs(int argc,char **argv);

    private:
        std::string mDir;
        std::string mPattern;
        std::string mOut;
};

cAppliFictObs::cAppliFictObs(int argc,char **argv) :
    mOut("_FObs")
{

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(mPattern,"Pattern of images"),
        LArgMain() << EAM (mOut,"Out",true,"Output file name")
    );
   #if (ELISE_windows)
        replace( mPattern.begin(), mPattern.end(), '\\', '/' );
   #endif

   SplitDirAndFile(mDir,mPattern,mPattern);

   cElemAppliSetFile anEASF(mPattern);
   const std::vector<std::string> * aSetName =   anEASF.SetIm();

   cNewO_NameManager *  aNM = NM(mDir);
   std::string aNameLTriplets = aNM->NameTopoTriplet(true);

   cXml_TopoTriplet  aLT = StdGetFromSI(aNameLTriplets,Xml_TopoTriplet);

   //pour chaque triplet recouper son elipse3d et genere les obs fict
   for (auto a3 : aLT.Triplets())
   {
       std::string  aName3R = aNM->NameOriOptimTriplet(true,a3.Name1(),a3.Name2(),a3.Name3());
       cXml_Ori3ImInit aXml3Ori = StdGetFromSI(aName3R,Xml_Ori3ImInit);


   }

   //cGenGaus3D aGG1(anEl);
   //std::vector<Pt3dr> aVP;

   //aGG1.GetDistribGaus(aVP,1+NRrandom3(2),2+NRrandom3(2),3+NRrandom3(2));

}


int FictiveObsFin_main(int argc,char ** argv)
{
    cAppliFictObs AppliFO(int argc,char ** argv);

    return EXIT_SUCCESS;
 
}

