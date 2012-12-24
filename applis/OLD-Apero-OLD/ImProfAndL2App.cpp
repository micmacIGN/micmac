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

#include "general/all.h"
#include "private/all.h"
#include "XML_GEN/all.h"
#include "im_tpl/image.h"

using namespace NS_ParamChantierPhotogram;


//    Ancienne syntaxe :
//    bin/ImProfAndL2App   ${ThisDir}   Key-Set-Hom-LFonGA  ${DirGA}GeoI/Nuage3DGeom-Im.tif  1 
//    KPL2A=Key-Assoc-Homol2Appuis
//
//
//    NOUVELLE SYNTAXE   
//       arg3 est soit fichier soit une cle de calcul a partir de arg
//
//    bin/ImProfAndL2App  Dir Key-Set-Hom  Key-Or-Nuage 
//


/*********************************************/
/*                                           */
/*            cAppliConv                     */
/*                                           */
/*********************************************/

class cAppliConvIP2L : public cAppliBatch
{
    public :
       cAppliConvIP2L(int argc,char ** argv);


       void Exec();
    private :
	int         mAppHIsP1;
        std::string         mNameNuage;
        cElNuage3DMaille *  mN3D;

        std::string mNameImProf;
        std::string mKeyPL2A;
        std::string mNameMasq;
};



cAppliConvIP2L::cAppliConvIP2L(int argc,char ** argv) :
   cAppliBatch
   (
        argc,argv,
        4,   // Trois argument obligatoires
	1,   // 1 Argument est un fichier
	""
   ) , 
   mAppHIsP1 (1),
   mNameNuage (argv[3]),
    mN3D       (0)
{

    ElInitArgMain
    (
           ARGC(),ARGV(),
           LArgMain() << EAM(mNameImProf) ,
           LArgMain() << EAM(mKeyPL2A,"KPL2A",true)
                      << EAM(mAppHIsP1,"HisP1",true) 
    );

}

void  cAppliConvIP2L::Exec()
{
   std::cout << CurF1() << "\n";
   std::string aNN  = ICNM()->StdCorrect(mNameNuage,CurF1(),true);
   std::cout <<"N3D : " << aNN << "\n";
   delete mN3D;
   mN3D = cElNuage3DMaille::FromFileIm(DirChantier()+aNN);

   ElPackHomologue aPackH = ElPackHomologue::FromFile(DirChantier()+CurF1());
// return;

   std::list<Appar23> aRes;

   for
   (
       ElPackHomologue::const_iterator itH = aPackH.begin();
       itH != aPackH.end();
       itH++
   )
   {
         Pt2dr aPHomApp = itH->P1();
	 Pt2dr aPIm = itH->P2();
	 if (! mAppHIsP1)
	    ElSwap(aPHomApp,aPIm);

         Pt2dr aPIndex = mN3D->Plani2Index(aPHomApp);

         if (mN3D->IndexHasContenuForInterpol(aPIndex))
         {
              Pt3dr aPTer = mN3D->PtOfIndexInterpol(aPIndex);
              aRes.push_back(Appar23(aPIm,aPTer));
	 }
   }

   std::string aNameAp;
   if (mKeyPL2A == "")
   {
       std::string aDir,aFile;
       SplitDirAndFile(aDir,aFile,CurF1());

        aNameAp = aDir + "Appuis" + aFile;
   }
   else
   {
        aNameAp = ICNM()->Assoc1To1(mKeyPL2A,CurF1(),true);
   }

   cListeAppuis1Im  aXmlApp = El2Xml(aRes);
   MakeFileXML<cListeAppuis1Im>
   (
        aXmlApp,
	DirChantier()+aNameAp
   );

   std::cout << "Appuis = " << aNameAp << "\n";


}

   //===========================================

int main(int argc,char ** argv)
{
    std::cout << "ARG[4] :  1 ->  P1 is Hom-Appuis \n";
    cAppliConvIP2L aAP(argc,argv);

    aAP.DoAll();

    // std::cout << aAP.CurCom() << "\n";



    return 0;
}







/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
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
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
