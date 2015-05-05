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
#include "im_tpl/image.h"




#define DEF_OFSET -12349876


class  cAppliSimul
{
    public :
        cAppliSimul(const std::string & aName,int argc,char ** argv);
    private :


        std::string  NameSauv(const std::string &);
        void Sauv(const std::string &,Im2D_REAL4 anIm,double aNivRand,bool ImGeom);
                    

        std::string         mNameIn;
        std::string         mNameManip;
        std::string         mDir;
        Tiff_Im             mTifIn;
        GenIm::type_el      mTypeIn;
        Pt2di               mSz;
        Im2D_REAL4          mImIn;
        TIm2D<float,double> mTImIn;
        Im2D_REAL4          mImOut;
        TIm2D<float,double> mTImOut;
        Im2D_REAL4          mPxX;
        TIm2D<float,double> mTPxX;
        Im2D_REAL4          mPxY;
        TIm2D<float,double> mTPxY;

        double              mResol;
        double              mPerX;
        double              mDerX;
        double              mPerY;
        double              mDerY;
        bool                mUsePxTransv;
        double              mNivRand;
        double              mKer;
};


std::string  cAppliSimul::NameSauv(const std::string & aName)
{
   return mDir + "SimEpip_" + mNameManip + "_" + aName + "_" + NameWithoutDir(mNameIn);
}

void cAppliSimul::Sauv(const std::string & aName,Im2D_REAL4 anIm,double aNivRand,bool isImGeom)
{
    std::string aNameOut = NameSauv(aName);
    GenIm::type_el aTypeOut =(isImGeom  ? GenIm::real4 : mTypeIn);

std::cout << aNameOut << " " << aTypeOut << "\n";

    Tiff_Im TiffOut  =   Tiff_Im 
                         (
                              aNameOut.c_str(),
                              mSz/mResol,
                              aTypeOut,
                              Tiff_Im::No_Compr,
			      Tiff_Im::BlackIsZero
                         );

    double aK = isImGeom ? 1.0 : mKer ;

    Fonc_Num aFIn = StdFoncChScale
                    (
                       anIm.in_proj(),
                       Pt2dr(0,0),
                       Pt2dr(mResol,mResol),
                       Pt2dr(aK,aK)  // Dilatation du noyau
                    );
     if (isImGeom) 
     {
        aFIn = aFIn/mResol;
     }
     if (aNivRand)
     {
            aFIn = aFIn + frandr() * aNivRand;
     }

     aFIn = Tronque(aTypeOut,aFIn);

     ELISE_COPY ( TiffOut.all_pts(), aFIn, TiffOut.out());

    // Fonc_Num aFIn = StdFoncChScale ()
}


cAppliSimul::cAppliSimul (const std::string & aName, int argc,char ** argv) :
    mNameIn ((aName=="-help") ? "data/TDM.tif" : aName),
    mDir    (DirOfFile(mNameIn)),
    mTifIn  (Tiff_Im::StdConv(mNameIn)),
    mTypeIn (mTifIn.type_el()),
    mSz     (mTifIn.sz()),
    mImIn   (mSz.x,mSz.y),
    mTImIn  (mImIn),
    mImOut  (mSz.x,mSz.y),
    mTImOut (mImOut),
    mPxX    (mSz.x,mSz.y),
    mTPxX   (mPxX),
    mPxY    (mSz.x,mSz.y),
    mTPxY   (mPxY),
    mResol  (3.578),
    mPerX   (0),
    mDerX   (-0.05),
    mPerY   (0),
    mDerY   (0.0),
    mUsePxTransv   (false),
    mNivRand       (0.0),
    mKer           (1.0)
{
  
   ElInitArgMain
   (
	argc,argv,
	LArgMain()  << EAM(mNameIn)
                    << EAM(mNameManip),
	LArgMain()  << EAM(mResol,"Resol",true)
                    << EAM(mPerX,"PerX",true)
                    << EAM(mDerX,"DerX",true)
                    << EAM(mPerY,"PerY",true)
                    << EAM(mDerY,"DerY",true)
                    << EAM(mNivRand,"Bruit",true)
                    << EAM(mKer,"Ker",true)
   );	

   mPerX *= mResol;
   mPerY *= mResol;

   ELISE_COPY(mTifIn.all_pts(),mTifIn.in(),mImIn.out());

   Fonc_Num  aFx = (mPerX >0) ?  (sin(FX/mPerX) *mPerX) : (FX-mSz.x/2.0);
   Fonc_Num  aFy = (mPerY >0) ?  (sin(FY/mPerY) *mPerY) : (FY-mSz.y/2.0);

   ELISE_COPY(mPxX.all_pts(), aFx *mDerX + aFy*mDerY, mPxX.out());


   if (!mUsePxTransv)
   {
      ELISE_COPY(mPxY.all_pts(),0.0,mPxY.out());
   }
     

   Pt2di aPOut;
   for (aPOut.x =0 ; aPOut.x < mSz.x ; aPOut.x++)
   {
       for (aPOut.y =0 ; aPOut.y < mSz.y ; aPOut.y++)
       {
           Pt2dr aPIn = Pt2dr(aPOut)+Pt2dr(mTPxX.get(aPOut),mTPxY.get(aPOut));
           mTImOut.oset(aPOut,mTImIn.getr(aPIn,0.0));
       }
   }

   Sauv("PxX",mPxX,0,true);
   Sauv("I2",mImIn,mNivRand,false);
   Sauv("I1",mImOut,mNivRand,false);

// void cAppliSimul::Sauv(const std::string & aName,Im2D_REAL4 anIm,double aNivRand,bool isImGeom)

/*
    Tiff_Im TiffOut  =     Tiff_Im 
                           (
                              aNameOut.c_str(),
                              aSz,
                              aType,
                              Tiff_Im::No_Compr,
			      tiff.phot_interp()
                          );


    Fonc_Num aFIn = StdFoncChScale
                 (
                       //aDebug ? ((FX/30)%2) && tiff.in_proj() : tiff.in_proj(),
                       aDebug ? tiff.in(0) : tiff.in_proj(),
                       Pt2dr(aP0.x,aP0.y),
                       Pt2dr(aScX,aScY),
                       aDilXY
                 );
*/
}



int main(int argc,char ** argv)
{
   ELISE_ASSERT(argc>=2,"Pas assez d'arg");
   cAppliSimul anAppli(argv[1],argc,argv);
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
