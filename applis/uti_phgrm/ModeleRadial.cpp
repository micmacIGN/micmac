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

bool BugMM = false;
using namespace NS_SuperposeImage;

/*
  Extrait un modele radial d'une image de points homologues
*/

namespace NB_ModelRadial
{

class cOneCoord
{
     public :
        cOneCoord
        (
            const std::string & aName,
	    double & aTr0,
            double aStep,
	    double aResol,
	    bool   aDequant,
	    bool   aACTR0,
	    double   aRoundTr0
        );
	Im2D_REAL8 Im() {return mIm;}
	Im2D_REAL8 ImR1() {return mImR1;}
	Pt2di SzR1() {return mSzR1;}
	Pt2dr SzRed() {return Pt2dr(mSzR1)/mResol;}
     private :
        Tiff_Im      mTif;
	double       mResol;
	double       mTr0;
	Pt2di        mSzR1;
	Pt2di        mSzRed;
	Im2D_REAL8   mIm;
	Im2D_REAL8   mImR1;
};


cOneCoord::cOneCoord
(
      const std::string & aName,
      double & aTr0,
      double aStep,
      double aResol,
      bool aDequant,
      bool aACTR0,
      double   aRoundTr0
)  :
   mTif    (Tiff_Im::StdConv(aName)),
   mResol  (aResol),
   mTr0    (aTr0),
   mSzR1   (mTif.sz()),
   mSzRed  (round_ni(Pt2dr(mSzR1)/mResol)),
   mIm     (mSzRed.x,mSzRed.y),
   mImR1   (mSzR1.x,mSzR1.y)
{
   ELISE_COPY(mImR1.all_pts(),mTif.in(0),mImR1.out());

  if (aDequant)
  {
      ElImplemDequantifier aDeq(mSzR1);
      aDeq.SetTraitSpecialCuv(true);
      aDeq.DoDequantif(mSzR1,mImR1.in(),true);

      ELISE_COPY(mImR1.all_pts(),aDeq.ImDeqReelle(),mImR1.out());
  }


   Pt2dr aPResol (mResol,mResol);
   Pt2dr aP0(0,0);
   std::cout << "ENTER \n";
   if (aACTR0)
   {
       double aSI,aS1;
       ELISE_COPY
       (
            mImR1.all_pts(),
           Virgule(1.0,mImR1.in(0)*aStep),
           Virgule(sigma(aS1),sigma(aSI))
       );
       mTr0 = aSI/aS1;
       std::cout << "COR TR = " << mTr0  << "\n";
       if (aRoundTr0!=0)
          mTr0 = aRoundTr0 * round_ni(mTr0/aRoundTr0);
       std::cout << "COR TR = " << mTr0  << "\n";
   }
   ELISE_COPY
   (
        mIm.all_pts(),
	StdFoncChScale((mImR1.in(0)*aStep -mTr0)*mTif.inside(),aP0,aPResol)
	/  StdFoncChScale(mImR1.inside(),aP0,aPResol),
	mIm.out()
   );
   std::cout << "DONE \n";
   aTr0 = mTr0;
}


class cModeleRadial
{
     public :
        cModeleRadial(const cGenereModeleRaster2Analytique & aGMR2A,
	              const std::string & aCh1,const std::string & aCh2);
	void ShowNorme();
	void ShowAngle();
	double FocaleInit ();
	void AllItere();
	void Sauv();
	void SauvImg(cDbleGrid & aG);
     private :

	 void OneItere();

         cGenereModeleRaster2Analytique mGMR2A;
         double            mResol;
         Pt2dr             mPas;
	 bool              mACTr0;
	 double            mRoundTr0;
	 Pt2dr             mTR0;
         const std::string mDir;
         const std::string mNameX;
         const std::string mNameY;

	 cOneCoord        mCX;
	 cOneCoord        mCY;
	 Pt2di            mSzR1;
	 Im2D_REAL8       mIx;
	 TIm2D<REAL8,REAL8> mTImX;
	 Im2D_REAL8       mIy;
	 TIm2D<REAL8,REAL8> mTImY;
	 Pt2di            mSzIm;
	 Pt2dr            mMil;
	 Fonc_Num         mFEc;
	 Fonc_Num         mFId;
	 double           mFocaleInit;
	 Pt2dr            mPPCInit;  // PP et Centre Dist sont confondu
	 ElDistRadiale_PolynImpair mDist;
	 cCamStenopeDistRadPol     mCam;
	 cSetEqFormelles  mSetEq;
	 cParamIFDistRadiale * mPIF;
	 cEqDirecteDistorsion * mEqD;

	 std::string            mCh1;
	 std::string            mCh2;

};


void cModeleRadial::OneItere()
{
  static int aCpt=-1; aCpt++;

  mSetEq.AddContrainte(mPIF->StdContraintes(),true);
  mSetEq.SetPhaseEquation();
  //  mSetEq.AddContrainte(); // ?? ROTATION CENTRE FIGE !!
  Pt2di aP;
  double aS1=0,aSEr=0;
  for (aP.x= 0; aP.x<mSzIm.x ; aP.x++)
  {
      for (aP.y=  0 ; aP.y<mSzIm.y ; aP.y++)
      {
	  BugMM =  (aP== Pt2di(0,0));
          Pt2dr aP2(mTImX.get(aP),mTImY.get(aP));
	  // aP2 = (Pt2dr(aP)-mPPCInit) * mFocaleInit;
	  // std::cout << mPPCInit << mFocaleInit << "\n";
	  Pt2dr aPR = Pt2dr(aP) * mResol;
          const std::vector<REAL> & aV= mEqD->AddObservation(aPR,aPR+aP2,1.0);
	  aS1++;
	  // std::cout << aV[0] << " " << aV[1] << aPR << aP2 << "\n";
	  for (int aK=0 ; aK<int(aV.size()) ; aK++)
	     aSEr += ElSquare(aV[aK]);
	   // std::cout << aSEr / aS1 << "\n";
	  // getchar();
      }
   }
   aSEr /= aS1;
   std::cout << "ERR = " << sqrt(aSEr) << "\n";
   mSetEq.SolveResetUpdate();
// getchar();
}

void cModeleRadial::AllItere()
{
   for (int aK=0 ; aK<3 ; aK++)
      OneItere();

    std::cout << "END Focale \n\n";

   for (int aD = 1 ; aD <= mGMR2A.DegPoly() ; aD++)
   {
        mPIF->SetDRFDegreFige(aD);
       for (int aK=0 ; aK<3 ; aK++)
          OneItere();
       std::cout << "END Degre " << aD <<  " \n\n";
   } 

   if ( mGMR2A.CLibre())
   {
       mPIF->SetCDistPPLie();
       std::cout << "Centre = " << mPIF->CurPP() << "\n";
       for (int aK=0 ; aK<6 ; aK++)
       {
          OneItere();
       }
       std::cout << "Centre = " << mPIF->CurPP() << "\n";
       std::cout << "END Centre  \n\n";
   }


/*
   if (1)
   {
       mPIF->SetLibertePPAndCDist(true,true);
       std::cout << "Centre = " << mPIF->CurPP() << "\n";
       for (int aK=0 ; aK<6 ; aK++)
       {
          OneItere();
       }
       std::cout << "Centre = " << mPIF->CurPP() << "\n";
       std::cout << "END Centre  \n\n";
   }
*/


}


double cModeleRadial::FocaleInit()
{
       double mSEc,mSId;
       ELISE_COPY
       (
           mIx.all_pts(),
           // Virgule(Hypot(mFEc),Hypot(mFId)),
           Virgule(Scal(mFEc,mFId),Scal(mFId,mFId)),
           Virgule(sigma(mSEc),sigma(mSId))
       );

       mFocaleInit =  mSEc / (mSId * mResol);
       std::cout << "FINIT := " << mFocaleInit << "\n";
/*
       Video_Win aW = Video_Win::WStd(mIx.sz(),1.0);
       ELISE_COPY 
       ( 
            aW.all_pts(), 
            TronkUC(128+100*(Hypot(mFEc)-Hypot(mFId)*mFocaleInit)),
            aW.ogray()
       );
       getchar();
*/
   
   return mFocaleInit;
}
/*
*/

void cModeleRadial::ShowNorme()
{

   Video_Win aW = Video_Win::WStd(mIx.sz(),1.0);
   double aMaxN;
   ELISE_COPY(aW.all_pts(),Hypot(mFEc),VMax(aMaxN));
   ELISE_COPY(aW.all_pts(),Hypot(mFEc)*(255.0/aMaxN),aW.ogray());
   getchar();
}


void cModeleRadial::ShowAngle()
{

   Video_Win aW = Video_Win::WStd(mIx.sz(),1.0);
   ELISE_COPY 
   ( 
         aW.all_pts(), 
	 polar(mFEc,0).v1() * (256 / (2*PI)),
	 aW.ocirc()
   );
   getchar();
}




std::vector<double> aNoAF;


cModeleRadial::cModeleRadial
(
     const cGenereModeleRaster2Analytique & aGMR2A,
     const std::string & aCh1,
     const std::string & aCh2
) :
   mGMR2A   (aGMR2A),
   mResol   (aGMR2A.SsResol()),
   mPas     (aGMR2A.Pas()),
   mACTr0   (aGMR2A.AutoCalcTr0().Val()),
   mRoundTr0(aGMR2A.RoundTr0().Val()),
   mTR0     (aGMR2A.Tr0().Val()),
   mDir     (aGMR2A.Dir()),

   // mNameX   ("Px1_Num6_DeZoom1_ChRouge_Vert_1.tif"),
   // mNameY   ("Px2_Num6_DeZoom1_ChRouge_Vert_1.tif"),
   mNameX   (aGMR2A.Im1()),
   mNameY   (aGMR2A.Im2()),
   mCX          (mDir+mNameX,mTR0.x,mPas.x,mResol,mGMR2A.Dequant(),mACTr0,mRoundTr0),
   mCY          (mDir+mNameY,mTR0.y,mPas.y,mResol,mGMR2A.Dequant(),mACTr0,mRoundTr0),
   mSzR1        (mCX.SzR1()),
   mIx          (mCX.Im()),
   mTImX        (mIx),
   mIy          (mCY.Im()),
   mTImY        (mIy),
   mSzIm        (mIx.sz()),
   mMil         (mCX.SzRed()/2.0),
   mFEc         (Virgule(mIx.in(),mIy.in())),
   mFId         (Virgule(FX-mMil.x,FY-mMil.y)),
   mFocaleInit  (FocaleInit()),
   mPPCInit     (mMil * mResol),
   mDist        (ElDistRadiale_PolynImpair::DistId(1.3*euclid(mPPCInit),mPPCInit,5)),
   mCam         (true,mFocaleInit,mPPCInit,mDist,aNoAF,0,mCX.SzR1()),
   mSetEq       (cNameSpaceEqF::eSysPlein,1000),
   mPIF         (mSetEq.NewIntrDistRad (true,&mCam,0)),
   mEqD         (mSetEq.NewEqDirecteDistorsion(*mPIF,cNameSpaceEqF::eTEDD_Bayer)),
   mCh1         (aCh1),
   mCh2         (aCh2)

{
     mPIF->SetFocFree(true);
     mPIF->SetLibertePPAndCDist(false,false);
     mSetEq.SetClosed();

}

void cModeleRadial::Sauv()
{
   if (! mGMR2A.SauvegardeMR2A().IsInit())
      return;

   ElDistortion22_Gen * aDistAnaly = mEqD->Dist(mTR0);

   Pt2dr aStepGr(mGMR2A.StepGridMR2A(),mGMR2A.StepGridMR2A());
   //aDistAnaly->SaveAsGrid(mGMR2A.NameSauvMR2A(),Pt2dr(0,0),mCX.SzR1(),aStepGr);

   cDbleGrid aDistGr
             (
	         true,
                 Pt2dr(0,0),
		 Pt2dr(mCX.SzR1()),
		 aStepGr,
		 *aDistAnaly
             );
	     

   cGridDirecteEtInverse aXMLGr = ToXMLExp(aDistGr);
   cBayerGridDirecteEtInverse aBGDI;
   aBGDI.Grid() = aXMLGr;
   aBGDI.Ch1() = mCh1;
   aBGDI.Ch2() = mCh2;
   MakeFileXML<cBayerGridDirecteEtInverse>(aBGDI,mGMR2A.NameSauvMR2A());


/*
    aDistGr.SaveXML(mGMR2A.NameSauv());
    cGridDirecteEtInverse aXMLGr = ToXMLExp(aDistGr);
    MakeFileXML<cGridDirecteEtInverse>(aXMLGr,mGMR2A.NameSauv());
*/

    SauvImg(aDistGr);
   delete aDistAnaly;
}

void cModeleRadial::SauvImg( cDbleGrid & aGR)
{
   if (! mGMR2A.SauvImgMR2A().IsInit())  
       return;
    std::string aDir,aName;
    SplitDirAndFile(aDir,aName,mGMR2A.SauvImgMR2A().Val());

    Im2D_REAL8 aImX(mSzR1.x,mSzR1.y);
    Im2D_REAL8 aImY(mSzR1.x,mSzR1.y);

    Pt2di aP;
    for  (aP.x=0; aP.x<mSzR1.x; aP.x++)
    {
        for  (aP.y=0; aP.y<mSzR1.y; aP.y++)
        {
	    Pt2dr aQ = aGR.Direct(Pt2dr(aP));
	    aImX.data()[aP.y][aP.x] = aQ.x;
	    aImY.data()[aP.y][aP.x] = aQ.y;
        }
    }
    std::cout<< aDir+"X_"+aName << "\n";


//  X et Y sont les images qui doivent ressembler a image en entree
    Tiff_Im::Create8BFromFonc
    (
         aDir+"X_"+aName,
	 mSzR1,
	 Max(0,Min(255,128 + (aImX.in()-FX)/mGMR2A.Pas().x))
    );

    Tiff_Im::Create8BFromFonc
    (
         aDir+"Y_"+aName,
	 mSzR1,
	 Max(0,Min(255,128 + (aImY.in()-FY)/mGMR2A.Pas().y))
    );



    Tiff_Im::Create8BFromFonc
    (
        aDir+"XResidu_"+aName,
        mSzR1,
        Max(0,Min(255,128 + (aImX.in()-FX-mCX.ImR1().in()*mPas.x)*100.0))
    );

    Tiff_Im::Create8BFromFonc
    (
        aDir+"YResidu_"+aName,
        mSzR1,
        Max(0,Min(255,128 + (aImY.in()-FY-mCY.ImR1().in()*mPas.y)*100.0))
    );

}


};

using namespace NB_ModelRadial;

int main(int argc,char ** argv)
{
    std::string aNameGMR2A = "GenereModeleRaster2Analytique";
    std::string  aNameFile;
    std::string  aTagAttrVal = aNameGMR2A;
    std::string aCh1,aCh2;

    ElInitArgMain
    (
         argc,argv,
	 LArgMain()  << EAM(aNameFile),
         LArgMain()  << EAM(aTagAttrVal,"Id",true)
	             << EAM(aCh1,"Ch1",true)
	             << EAM(aCh2,"Ch2",true)
    );

    ELISE_ASSERT(argc>=2,"Not Enough arg");
    cGenereModeleRaster2Analytique  aGMR2A = StdGetObjFromFile<cGenereModeleRaster2Analytique>
                                      (
                                         aNameFile,
                                         "include/XML_GEN/SuperposImage.xml",
					 aTagAttrVal,
                                         aNameGMR2A,
					 (aTagAttrVal!=aNameGMR2A)
                                      );
    cModeleRadial aMR(aGMR2A,aCh1,aCh2);
    
    // aMR.ShowAngle();
    // aMR.ShowNorme();

    aMR.AllItere();
    aMR.Sauv();

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
