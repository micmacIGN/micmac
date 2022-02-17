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

#include "all_etal.h"
#include "XML_GEN/all.h"


class cOneImSaisie
{
   public :
      cOneImSaisie
      (
          int   aDecX,
          const std::string & aNameImPol,
          const std::string & aNamePointePol,
	  const cPolygoneEtal & aPol
      ) :
        mTifIm   (Tiff_Im::StdConv(aNameImPol)),
	mSz      (mTifIm.sz()),
	mDecX    (aDecX),
	mPointes (aPol,aNamePointePol)
      {
      }

      void Sauv
           (
	       Tiff_Im aTifGlob,
               FILE * aFp
	   )
      {
           ELISE_COPY
	   (
	       rectangle(Pt2di(mDecX,0),Pt2di(mDecX+mSz.x,mSz.y)),
	       trans(mTifIm.in(),Pt2di(-mDecX,0)),
	       aTifGlob.out()
	   );
	   for 
	   (
	      std::list<cPointeEtalonage>::iterator itPE = mPointes.Pointes().begin();
	      itPE != mPointes.Pointes().end();
	      itPE++
	   )
	   {
	      Pt2dr aPim = itPE->PosIm();
	      fprintf
	      (
	          aFp,
		  "%d %f %f\n",
		  itPE->Cible().Ind(),
		  aPim.x+mDecX,
		  aPim.y
              );
           }
      }


      Tiff_Im        mTifIm;
      Pt2di          mSz;
      int            mDecX;
      cSetPointes1Im mPointes;
};


class AllImSaisie
{
    public :
       AllImSaisie
       (
           const std::string &  aDir,
           const std::string &  aNamePol
       )  :
          mSzX (0),
          mSzY (0),
          mDir (aDir),
	  mPol (cPolygoneEtal::FromName(mDir+aNamePol))
       {
       }

       void AddOneImSaisie
            (
                const std::string & aNameImPol,
                const std::string & aNamePointePol
            ) 
       {
          mIS.push_back
	  (
	      cOneImSaisie
	      (
	          mSzX,
	          mDir+aNameImPol,
	          mDir+aNamePointePol,
		  *mPol
	      )
	  );
	  mSzX += mIS.back().mSz.x;
	  ElSetMax(mSzY,mIS.back().mSz.y);
       }

       void Sauv(const std::string & aName)
       {
          std::string aNameTif = mDir+aName+".tif";
	  Tiff_Im aTif
	          (
		      aNameTif.c_str(),
		      Pt2di(mSzX,mSzY),
		      GenIm::u_int1,
		      Tiff_Im::No_Compr,
		      Tiff_Im::RGB
                  );
          std::string aNamePointes = mDir+aName+".txt";
          FILE * aFp = ElFopen(aNamePointes.c_str(),"w");

	  for (int aK=0 ; aK<int(mIS.size()) ; aK++)
	     mIS[aK].Sauv(aTif,aFp);

	  ElFclose(aFp);
       }

    private :
       int                       mSzX;
       int                       mSzY;
       std::string               mDir;
       cPolygoneEtal *           mPol;
       std::vector<cOneImSaisie> mIS;
};




void CatCDD()
{
  if (0)
  {
      AllImSaisie anAIS
             (
	        "/DATA2/Calibration/References/PolygCapDieux/",
		"PolyCapriceDesDieux.xml"
	     );
  // anAIS.AddOneImSaisie("Image4.tif","Image4.txt");
      anAIS.AddOneImSaisie("Image3.tif","Image3.txt");
      anAIS.AddOneImSaisie("Image2.tif","Image2.txt");
      anAIS.AddOneImSaisie("Image1.tif","Image1.txt");

      anAIS.Sauv("Image4321");
  }
  if (1)
  {
      AllImSaisie anAIS
             (
	        "/mnt/data/Calib/References/PolygCapDieux/",
		"PolCDDCompl.xml"
	     );
  // anAIS.AddOneImSaisie("Image4.tif","Image4.txt");
      anAIS.AddOneImSaisie("Image5Clous.tif","Image5Clous.txt");
      anAIS.AddOneImSaisie("Image2.tif","Image2.txt");
      anAIS.AddOneImSaisie("Image1.tif","Image1.txt");

      anAIS.Sauv("Image521");
  }
}

int main(int argc,char ** argv)
{
   CatCDD();
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
