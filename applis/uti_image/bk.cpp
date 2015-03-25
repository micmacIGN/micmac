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


class cBookMaker 
{

    public :
       cBookMaker(int argc, char ** argv)  :
          mPrefBK ("BK")
       {
             ElInitArgMain
             (
                 argc,argv,
                 LArgMain() 	<< EAM(mDir) 
		                << EAM(mPat)
				<< EAM(aRot),
                 LArgMain() 	<< EAM(mPrefBK,"BK_",true)
             );	
	      
	     mFiles = ListFileMatch(mDir,mPat,1);
       }
       void DoIt()
       {
           for 
           (
               std::list<std::string>::iterator iT = mFiles.begin();
               iT != mFiles.end();
	       iT++
           )
           {
               DoIt1(*iT);
           }
       }
       void DoIt1(const std::string & aName);

    private :
       INT                     aRot;
       std::string             mDir;
       std::string             mPat;
       std::string             mPrefBK;
       std::list<std::string>  mFiles;
};


void cBookMaker::DoIt1(const std::string & aName)
{
   Tiff_Im aFile(aName.c_str());
   Pt2di aSz = aFile.sz();
   Im2D_U_INT1 anImIn(aSz.x,aSz.y);
   ELISE_COPY
   (
       anImIn.all_pts(),
       aFile.in(),
       anImIn.out()
   );

   std::string aNameOut 
	     = mDir + mPrefBK + aName.substr(mDir.size(),aName.size());
   cout << aName << "=>" << aNameOut << "\n";
   
   Pt2di aSzOut = aSz;
   if (mod(aRot,2)==1)
       aSzOut = Pt2di(aSz.y,aSz.x);

   Fonc_Num fChC = Virgule(FX,FY);
   if (mod(aRot,4)==1)
      fChC = Virgule(FY,aSzOut.x-FX-1);   
   if (mod(aRot,4)==3)
      fChC = Virgule(aSzOut.y-FY-1,FX);   
   if (mod(aRot,4)==2)
      fChC = Virgule(aSzOut.x-FX-1,aSzOut.y-FY-1);   

   Tiff_Im aFileOut
	   (
	       aNameOut.c_str(),
	       aSzOut,
	       GenIm::bits1_msbf,
	       Tiff_Im::Group_4FAX_Compr,
	       Tiff_Im::BlackIsZero
	   );
   ELISE_COPY(aFileOut.all_pts(),anImIn.in()[fChC]>128,aFileOut.out());
   cout  << "Done " << aNameOut << "\n";
}


int main(int argc,char ** argv)
{
    string Name;


    cBookMaker aBM(argc,argv);
    aBM.DoIt();
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
