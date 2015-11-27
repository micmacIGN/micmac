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
#include <algorithm>
#include "im_tpl/image.h"
#include "im_special/hough.h"



static std::string TheDir = "/media/MYPASSPORT/Documents/Cours/ENSG-TI-IT2/Geom/";
static int TheScale = 5;


class cTestGeom
{
    public :

        cTestGeom (const std::string & aName) :
            mName       (aName),
            mNameFull   (CalcNameResult("")),
            mFile       (Tiff_Im::StdConv(mNameFull)),
            mSz         (mFile.sz()),
            mIm         (mSz.x,mSz.y),
            mTIm        (mIm) ,
            mImRotate   (mSz.x,mSz.y),
            mTIR        (mImRotate) 
        {
            ELISE_COPY(mFile.all_pts(),mFile.in(),mIm.out());
        }

        void TestInterp ( cInterpolateurIm2D<double> *,const std::string & aName);
        void TurnOneTeta(int aScale,double aTeta);
        void ReducSquare();
        std::string CalcNameResult(const std::string& aStep)
        {
            return  TheDir+StdPrefix(mName) + aStep + ".tif";
        }

        std::string  mName;
        std::string  mNameFull;
        Tiff_Im      mFile;
        Pt2di        mSz;

        Im2D_REAL8            mIm;
        TIm2D<double,double>  mTIm;

        Im2D_REAL8            mImRotate;
        TIm2D<double,double>  mTIR;

        cInterpolateurIm2D<double>  * mInterp;
        std::string                   mNameInt;
};



void cTestGeom::TurnOneTeta(int  aScale,double aTeta)
{
   Pt2di aSzS = mSz/ aScale;
   Im2D_REAL8            aItmp(aSzS.x,aSzS.y,0.0);
   TIm2D<double,double>  aTItmp(aItmp);

   ElSimilitude aS = ElSimilitude::SimOfCentre(mSz/2.0,Pt2dr::FromPolar(1.0,aTeta));

   Pt2di aP;
   int aSzK = mInterp->SzKernel();
   Pt2di aQ0 (aSzK+1,aSzK+1);
   Pt2di aQ1 = mSz - aQ0;
   Box2di aBoxK(aQ0,aQ1);

   for (aP.x=0 ; aP.x<aSzS.x ; aP.x++)
   {
      for (aP.y=0 ; aP.y<aSzS.y ; aP.y++)
      {
           Pt2dr aQ = aS(aP) * aScale;
           if (aBoxK.inside(aQ))
               aTItmp.oset(aP,mInterp->GetVal(mImRotate.data(),aQ));
      }
   }
   ELISE_COPY(aItmp.all_pts(),aItmp.in(),mImRotate.out());

}


void cTestGeom::TestInterp
     (
        cInterpolateurIm2D<double> * anInterp,
        const std::string & aNameInt
     )
{
    mInterp = anInterp;
    mNameInt = aNameInt;

    ELISE_COPY(mIm.all_pts(),mIm.in(),mImRotate.out());
/*
    int aNbTeta = 20;

    for (int aK=0 ; aK< 2*aNbTeta ; aK++)
    {
std::cout << "K= " << aK << "\n";
        TurnOneTeta(1.0,(aK<aNbTeta) ? 0.1 :-0.1 );
         if ((aK==0) || (aK==1) || (aK==( 2*aNbTeta-1)))
         {
            Tiff_Im::Create8BFromFonc
            (
                CalcNameResult("_"+mNameInt+"_"+ToString(aK)),
                mSz, 
                mImRotate.in()
            );
         }
    }
*/


    int aScale=TheScale;

     TurnOneTeta(aScale,0);
     Tiff_Im::Create8BFromFonc
     (
          CalcNameResult("_Scale_"+mNameInt+"_"+ToString(aScale)),
          mSz/aScale, 
          mImRotate.in()
     );
/*
*/
}


void cTestGeom::ReducSquare()
{
   Pt2di aSzS = mSz/ TheScale;
   Im2D_REAL8  aItmp(aSzS.x,aSzS.y,0.0);
   ELISE_COPY
   (
        mIm.all_pts(),
        mIm.in(),
        aItmp.histo(true).chc(Virgule(FX/TheScale,FY/TheScale))
   );
   Tiff_Im::Create8BFromFonc
   (
      CalcNameResult("_ScaleCarre_"+ToString(TheScale)),
      mSz/TheScale, 
      aItmp.in() / ElSquare(TheScale)
   );
}




void TestGeom(const std::string & aName)
{
    cTestGeom aTG(aName);

    aTG.ReducSquare();
    // aTG.TestInterp(new cInterpolSinusCardinal<double>(7,false),"SinCard");

    
    aTG.TestInterp(new cInterpolBilineaire<double>,"Bilin");
    aTG.TestInterp(new cInterpolBicubique<double>(-0.5),"BiCub");
    aTG.TestInterp(new cInterpolPPV<double>,"PPV");

}



int main(int argc,char ** argv)
{

    // TestReco("0",3.0,false); 
    // TestReco("00",3.0,true); 
    // TestReco("1",3.0); 
    // TestReco("2",3.0,false); 
    // TestReco("3",3.0,false); 
     TestGeom("HieroInit.tif"); 

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
