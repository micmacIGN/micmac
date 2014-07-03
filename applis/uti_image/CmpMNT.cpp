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

Im2D_Bits<1> MasqStd()
{
   Im2D_Bits<1> aRes(1101,1301);

   Pt2dr aP1(230,0);
   Pt2dr aP2(1100,188);

   Pt2dr aNorm = (aP2-aP1) * Pt2dr(0,1);

   ELISE_COPY
   (
       aRes.all_pts(),
       ((FX-aP1.x)*aNorm.x + (FY-aP1.y) * aNorm .y) > 0,
       aRes.out()
   );
   // Tiff_Im::Create8BFromFonc("Toto.tif",aRes.sz(),255*aRes.in());
   // ELISE_COPY(aRes.all_pts(),aRes.in(),Video_Win::WStd(aRes.sz()/2,0.5).odisc());

   return aRes;

}

struct cStat
{
    public :

      cStat (double aMinDif,double aMaxDif,
              Im2D_REAL8 IRef,Im2D_REAL8 ITest,
	      Fonc_Num FMasq);
      void Show(int aDelta);

      int        mMinDif;
      int        mMaxDif;
      int        mNbDif;
      Im1D_REAL8 mHistoDif;
      double *   mDH;
      Im1D_REAL8 mCumHisto;
      double *   mDC;
      double     mFPerC;
      double     mPop;
      double     mSom;
      double     mSom2;
      double     mCorrelPente;

};


cStat::cStat 
(
    double aMinDif,
    double aMaxDif,
    Im2D_REAL8  aIRef,
    Im2D_REAL8  aITest,
    Fonc_Num FMasq
) :
   mMinDif   (round_down(aMinDif)-1),
   mMaxDif   (round_up(aMaxDif)+1),
   mNbDif    (mMaxDif-mMinDif+1),
   mHistoDif (mNbDif,0.0),
   mDH       (mHistoDif.data()),
   mCumHisto (mNbDif,0.0),
   mDC       (mCumHisto.data())
{



   ELISE_COPY
   (
       select(aIRef.all_pts(),FMasq).chc(round_ni(aIRef.in()-aITest.in())-mMinDif),
       1,
       mHistoDif.histo()
   );

  
   mDC[0] = mDH[0];
   for (int aV=1 ; aV<mNbDif ; aV++)
   {
       mDC[aV] =  mDC[aV-1] +  mDH[aV];
   }
   mPop =  mDC[mNbDif-1];
   mFPerC =mPop / 100.0;

   mSom = 0;
   mSom2 = 0;
   for (int aV=0 ; aV<mNbDif ; aV++)
   {
      mSom += (aV+mMinDif) * mDH[aV];
      mSom2 += ElSquare(aV+mMinDif) * mDH[aV];
   }
   mSom /= mPop;
   mSom2 /= mPop;
   mSom2 -= ElSquare(mSom);

   {
       double aFD = 1.0;
       Pt2di aSz = aIRef.sz();
       Im2D_Bits<1> aMasq(aSz.x,aSz.y);
       ELISE_COPY(aMasq.all_pts(),FMasq,aMasq.out());
       Im2D_REAL8 aGxR(aSz.x,aSz.y);
       Im2D_REAL8 aGyR(aSz.x,aSz.y);
       ELISE_COPY
       (
           aIRef.all_pts(),
           deriche(aIRef.in_proj(),aFD),
           Virgule(aGxR.out(),aGyR.out())
       );

       Im2D_REAL8 aGxT(aSz.x,aSz.y);
       Im2D_REAL8 aGyT(aSz.x,aSz.y);
       ELISE_COPY
       (
           aIRef.all_pts(),
           deriche(aITest.in_proj(),aFD),
           Virgule(aGxT.out(),aGyT.out())
       );

       TIm2DBits<1> aTM(aMasq);
       TIm2D<REAL8,REAL8> aTGxR(aGxR);
       TIm2D<REAL8,REAL8> aTGyR(aGyR);
       TIm2D<REAL8,REAL8> aTGxT(aGxT);
       TIm2D<REAL8,REAL8> aTGyT(aGyT);

       Mat_Inertie<Pt2dr> aMat;
       Pt2di aP;
       for (aP.x=0 ; aP.x<aSz.x; aP.x++)
       {
           for (aP.y=0 ; aP.y<aSz.y; aP.y++)
	   {
	      if (aTM.get(aP))
	      {
	          aMat.add_pt_en_place
		  (
		     Pt2dr(aTGxR.get(aP),aTGyR.get(aP)),
		     Pt2dr(aTGxT.get(aP),aTGyT.get(aP))
		  );
	      }
	   }
       }
       mCorrelPente = aMat.correlation();
   }
   std::cout << "Moy = " << mSom  << " ; ECt = " << sqrt(mSom2) 
             << " ; Pente Correl = " << mCorrelPente << "\n";
}

void cStat::Show(int aDelta)
{
   for 
   (
        int aD=ElMax(0,-mMinDif-aDelta);
        aD < ElMin(mNbDif-1,-mMinDif+aDelta);
	aD++
   )
   {
      std::cout << "D,C,H "
                <<  aD+mMinDif <<  " # "
                <<  (mDC[aD]/mFPerC)  <<  " # "
                <<  (mDH[aD]/mFPerC)  <<  " \n" ;
   }
}





int main(int argc,char ** argv)
{
   bool BDT = true;
   double aPasRef = (BDT ? 10 : 5);
   std::string aDirGlob = "/DATA2/JeuxReferences/HRS-Manosque/";
   std::string aDirRef = aDirGlob+ "REF/";
   // std::string aDirTest = aDirGlob+ "ResArticle/MaxOfScore/";
   // std::string aDirTest = aDirGlob+ "ResArticle/CoxRor/";
   std::string aDirTest = aDirGlob+"ResArticle/PrgDyn8Dir-Exp_2_1-Quad050/";

   // std::string  aNameRef = aDirRef+ "LASERDTM(m).HDR";
   // std::string  aNameTest = aDirTest+ "Bascule-Laser.tif";
   // std::string  aNameDif = aDirTest+ "ImDif.tif";
   std::string  aNameRef = aDirRef+   (BDT ? "BDTOPO.HDR" : "LASERDTM(m).HDR");
   std::string  aNameTest = aDirTest+ (BDT ? "Bascule-BDTopo.tif" : "Bascule-Laser.tif");
   std::string  aNameDif = aDirTest+ "ImDif-BdTopo.tif";

   Tiff_Im aFileRef = Tiff_Im::StdConv(aNameRef);
   Tiff_Im aFileTest = Tiff_Im::StdConv(aNameTest);

   Pt2di aSzRes = Inf(aFileRef.sz(),aFileTest.sz());

   
   Im2D_REAL8 aITest(aSzRes.x,aSzRes.y);
   ELISE_COPY(aITest.all_pts(),aFileTest.in(),aITest.out());

   Im2D_REAL8 aIRef(aSzRes.x,aSzRes.y);
   ElImplemDequantifier aDeq(aSzRes);
   aDeq.SetTraitSpecialCuv(true);
   aDeq.DoDequantif(aSzRes,aFileRef.in(),true);
   ELISE_COPY ( aIRef.all_pts(), aDeq.ImDeqReelle(), aIRef.out());


   Im2D_Bits<1> aMasq= MasqStd(); 

   Fonc_Num aFoncDif = aIRef.in()-aITest.in();

   double aMinDif,aMaxDif;
   ELISE_COPY
   (
       select(aIRef.all_pts(),aMasq.in()),
       Max(-1e5,Min(1e5, aFoncDif)),
       VMin(aMinDif) | VMax(aMaxDif)
   );


   Im2D_REAL8 aImPente(aSzRes.x,aSzRes.y);
   ELISE_COPY
   (
      aImPente.all_pts(),
      polar(deriche(aIRef.in_proj(),1.0),0).v0()/aPasRef,
      aImPente.out()
   );

   Video_Win aW = Video_Win::WStd(aSzRes,0.5);
   ELISE_COPY
   (
       select(aImPente.all_pts(),aImPente.in()<0.4),
       P8COL::blue,
       aW.odisc()
    );
   ELISE_COPY
   (
       select(aImPente.all_pts(),aImPente.in()<0.2),
       P8COL::yellow,
       aW.odisc()
   );
   /*
   ELISE_COPY
   (
       aImPente.all_pts(),
       Min(255,aImPente.in()*255),
       aW.ogray()
    );
   */

    std::cout << "DIF IN " << aMinDif << " : " << aMaxDif << "\n";
   
   cStat aStglob(aMinDif,aMaxDif,aIRef,aITest,aMasq.in());


   cStat aStat0(aMinDif,aMaxDif,aIRef,aITest,aMasq.in() && (aImPente.in()<0.2));
   cStat aStat1(aMinDif,aMaxDif,aIRef,aITest,aMasq.in() && (aImPente.in()>0.2) && (aImPente.in()<0.4));
   cStat aStat2(aMinDif,aMaxDif,aIRef,aITest,aMasq.in() && (aImPente.in()>0.4));

   // aStglob.Show(20);
   Tiff_Im aFileDif
            (
	        aNameDif.c_str(),
		Inf(aFileRef.sz(),aFileTest.sz()),
		GenIm::u_int1,
		Tiff_Im::No_Compr,
		Tiff_Im::BlackIsZero
	    );

   ELISE_COPY
   (
        aFileDif.all_pts(),
	Max(0,Min(255,128+aFileRef.in()-aFileTest.in())),
	aFileDif.out()
   );
   getchar();
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
