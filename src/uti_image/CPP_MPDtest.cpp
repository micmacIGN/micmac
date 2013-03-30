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
// #include "anag_all.h"

/*
void f()
{
    FILE * aFP = ElFopen(MMC,"w");
    ElFclose(aFP);
}

*/

#include "StdAfx.h"

#if (ELISE_X11)



using namespace NS_ParamChantierPhotogram;

#if (0)


#endif





Fonc_Num Moy(Fonc_Num aF,int aNb)
{
   return rect_som(aF,aNb) / ElSquare(1.0+2*aNb);
}

Fonc_Num Moy1(Fonc_Num aF) {return Moy(aF,1);}


Fonc_Num MoyIter(Fonc_Num aF,int aNb,int aNbIter)
{
   for (int aK=0; aK<aNbIter ; aK++)
       aF = Moy(aF,aNb);

   return aF;
}



Fonc_Num Correl(Fonc_Num aF1,Fonc_Num aF2,int aNb)
{
   Symb_FNum aM1 (Moy(aF1,aNb));
   Symb_FNum aM2 (Moy(aF2,aNb));

   Fonc_Num aEnct1 = Moy(Square(aF1),aNb) -Square(aM1);
   Fonc_Num aEnct2 = Moy(Square(aF2),aNb) -Square(aM2);


   return (Moy(aF1*aF2,aNb)  -aM1*aM2) / sqrt(Max(1e-5,aEnct1*aEnct2));
}

void AutoCorrel(const std::string & aName)
{
   Tiff_Im aTF(aName.c_str());
   Pt2di aSz = aTF.sz();
   Im2D_REAL4 anI(aSz.x,aSz.y);
   ELISE_COPY(aTF.all_pts(),aTF.in(),anI.out());

   int aNb = 2;

   Fonc_Num aF = 1.0;
   for (int aK=0 ; aK<4 ; aK++)
   {
      aF = Min(aF,Correl(anI.in(0),trans(anI.in(0),TAB_4_NEIGH[aK])*(aNb*2),aNb));
   }
  
   Tiff_Im::Create8BFromFonc
   (
       StdPrefix(aName)+"_AutoCor.tif",
       aSz,
       Min(255,Max(0,(1+aF)*128))
   );
}


Im2D_REAL4 Conv2Float(Im2DGen anI)
{
   Pt2di aSz = anI.sz();
   Im2D_REAL4 aRes(aSz.x,aSz.y);
   ELISE_COPY(anI.all_pts(),anI.in(),aRes.out());
   return aRes;
}


/*
Im2D_REAL4 RecursiveImpaint
     (
          Im2D_REAL4 aFlMaskInit,
          Im2D_REAL4 aFlMaskFinal,
          Im2D_REAL4 aFlIm,
          int        aDeZoom,
          int        aZoomCible
     );

template <class TypeIn,class TypeOut> 
Im2D<TypeIn,TypeOut> ImpaintL2
     (
         Im2D_Bits<1>           aB1MaskInit,
         Im2D_Bits<1>           aB1MaskFinal,
         Im2D<TypeIn,TypeOut>   anIn
     )
{
   Im2D_REAL4  aFlRes = RecursiveImpaint
                         (
                             Conv2Type(aB1MaskInit,(Im2D_REAL4*)0),
                             Conv2Type(aB1MaskFinal,(Im2D_REAL4*)0),
                             Conv2Type(anIn,(Im2D_REAL4*)0),
                             1,
                             16
                         );


   return Conv2Type(aFlRes,(Im2D<TypeIn,TypeOut>*)0);
}
*/



void TestKL()
{
   Pt2di aSZ(200,200);
   Im2D_Bits<1> aImMasqF(aSZ.x,aSZ.y,1);

   Im2D_Bits<1> aImMasqDef(aSZ.x,aSZ.y,1);
   ELISE_COPY(rectangle(Pt2di(70,0),Pt2di(130,200)),0,aImMasqDef.out());

   Im2D<U_INT2,INT> aImVal(aSZ.x,aSZ.y);
   ELISE_COPY(aImVal.all_pts(),FX,aImVal.out());

   Video_Win aW=Video_Win::WStd(aSZ,3.0);
   ELISE_COPY(aW.all_pts(),aImVal.in(),aW.ogray());
   ELISE_COPY(aW.all_pts(),aImMasqDef.in(),aW.odisc());
   getchar();


   aImVal = ImpaintL2(aImMasqDef,aImMasqF,aImVal);

   // NComplKLipsParLBas(aImMasqDef,aImMasqF,aImVal,1.0);

   ELISE_COPY(aW.all_pts(),aImVal.in(),aW.ogray());

   Tiff_Im::Create8BFromFonc("toto.tif",aSZ,aImVal.in());
   getchar();
}
#if (0)
#endif







void TestMultiEch_Gauss(int argc,char** argv)
{
   std::string aNameIm;
   Pt2di aP0(0,0),aSz;


   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(aNameIm,"Name Im"),
        LArgMain()  << EAM(aP0,"P0",true,"")
                    << EAM(aSz,"Sz",true,"")
   );

   Tiff_Im aTF = Tiff_Im::StdConvGen(aNameIm,1,false);
   if (! EAMIsInit(&aSz))
   {
      aSz = aTF.sz();
   }
   Video_Win  aW = Video_Win::WStd(aSz,1.0);

   Im2D_REAL4 anImOri(aSz.x,aSz.y);
   Im2D_REAL4 aGMax(aSz.x,aSz.y,-1);
   Im2D_INT4 aKMax(aSz.x,aSz.y,-1);
   ELISE_COPY
   (
        anImOri.all_pts(),
        trans(aTF.in(0),aP0),
        anImOri.out()
   );

   std::vector<Im2D_REAL4> aVG;
   for (int aK=0 ; aK< 100 ; aK++)
   {
       Im2D_REAL4 anI(aSz.x,aSz.y);
       // TIm2D<REAL4,REAL8> aTIm(anI);
       ELISE_COPY(anI.all_pts(),anImOri.in(),anI.out());
       double aSigm = aK;
       double aSigmM = aK+1;

       if (aSigm)
          FilterGauss(anI,aSigm);

       Im2D_REAL4 anI2(aSz.x,aSz.y);
       // TIm2D<REAL4,REAL8> aTIm2(anI2);
       ELISE_COPY(anI.all_pts(),Square(anI.in()),anI2.out());

       FilterGauss(anI,aSigmM);
       FilterGauss(anI2,aSigmM);
       

       double aSom;
       Im2D_REAL4 aImEc(aSz.x,aSz.y);
       ELISE_COPY(aW.all_pts(),sqrt(anI2.in()-Square(anI.in())),aImEc.out()|sigma(aSom));
       aSom /= aSz.x*aSz.y;
       ELISE_COPY(aImEc.all_pts(),aImEc.in() *(1.0/(aSom*(10+aK))),aImEc.out());

       ELISE_COPY(select(aImEc.all_pts(),aImEc.in()>aGMax.in()),Virgule(aImEc.in(),aK),Virgule(aGMax.out(),aKMax.out()));
       ELISE_COPY(aW.all_pts(),Min(255,128*pow(aImEc.in(),0.5)),aW.ogray());
       std::cout << "AAAAAAaaaa   " << aSom << " " << aK << "\n" ;  
   }
   ELISE_COPY(aW.all_pts(),aKMax.in(),aW.ogray());
   Tiff_Im::Create8BFromFonc("Scale.tif",aSz,aKMax.in());
   getchar();
}

Fonc_Num sobel_0(Fonc_Num f)
{
    Im2D_REAL8 Fx
               (  3,3,
                  " -1 0 1 "
                  " -2 0 2 "
                  " -1 0 1 "
                );
    Im2D_REAL8 Fy
               (  3,3,
                  " -1 -2 -1 "
                  "  0  0  0 "
                  "  1  2  1 "
                );
   return
       Abs(som_masq(f,Fx,Pt2di(-1,-1)))
     + Abs(som_masq(f,Fy));
}


void TestMultiEch_Gauss2(int argc,char** argv)
{
   std::string aNameIm;
   Pt2di aP0(0,0),aSz;

   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(aNameIm,"Name Im"),
        LArgMain()  << EAM(aP0,"P0",true,"")
                    << EAM(aSz,"Sz",true,"")
   );

   Tiff_Im aTF = Tiff_Im::StdConvGen(aNameIm,1,false);
   if (! EAMIsInit(&aSz))
   {
      aSz = aTF.sz();
   }
   Video_Win  aW = Video_Win::WStd(aSz,1.0);

   Im2D_REAL4 anImOri(aSz.x,aSz.y);
   Im2D_REAL4 aGMax(aSz.x,aSz.y,-1);
   Im2D_INT4 aKMax(aSz.x,aSz.y,-1);
   ELISE_COPY
   (
        anImOri.all_pts(),
        trans(aTF.in(0),aP0),
        anImOri.out()
   );

   std::vector<Im2D_REAL4> aVG;
   for (int aK=0 ; aK< 100 ; aK++)
   {
       Im2D_REAL4 anI(aSz.x,aSz.y);
       // TIm2D<REAL4,REAL8> aTIm(anI);
       ELISE_COPY(anI.all_pts(),anImOri.in(),anI.out());
       double aSigm = aK;
       // double aSigmM = aK+1;

       if (aSigm)
          FilterGauss(anI,aSigm);

       double aSom;
       Im2D_REAL4 aImEc(aSz.x,aSz.y);
       ELISE_COPY(aW.all_pts(),sobel_0(anI.in_proj()),aImEc.out()|sigma(aSom));

       aSom /= aSz.x*aSz.y;
       ELISE_COPY(aImEc.all_pts(),aImEc.in() *(1.0/(aSom*(1+0.0*aK))),aImEc.out());

       ELISE_COPY(select(aImEc.all_pts(),aImEc.in()>aGMax.in()),Virgule(aImEc.in(),aK),Virgule(aGMax.out(),aKMax.out()));
       ELISE_COPY(aW.all_pts(),Min(255,128*pow(aImEc.in(),0.5)),aW.ogray());
       std::cout << "AAAAAAaaaa   " << aSom << " " << aK << "\n" ;  
   }
   ELISE_COPY(aW.all_pts(),aKMax.in(),aW.ogray());
   Tiff_Im::Create8BFromFonc("Scale.tif",aSz,aKMax.in());
   getchar();
}

Fonc_Num GradBasik(Fonc_Num f)
{
    Im2D_REAL8 Fx
               (  3,3,
                  " 0  0 0 "
                  " 0 -1 1 "
                  " 0  0 0 "
                );
    Im2D_REAL8 Fy
               (  3,3,
                  "  0  0  0 "
                  "  0 -1  0 "
                  "  0  1  0 "
                );
   return (Abs(som_masq(f,Fx)) + Abs(som_masq(f,Fy))) / 4.0;
}

Fonc_Num  MoyRect(Fonc_Num aF,int aSzV)
{
   Fonc_Num  aRes = 0;
   for (int aK=0 ; aK< 9 ; aK++)
       aRes = aRes + trans(aF,TAB_9_NEIGH[aK]*aSzV);
    return aRes / 9;
}


Fonc_Num  EcartType(Fonc_Num aF,int aSzV)
{
   Fonc_Num  aRes = MoyRect(Square(aF),aSzV) -Square(MoyRect(aF,aSzV));

   return sqrt(Max(0,aRes));
}



template <class Type,class TypeBase> class cCalcSzWCorrel
{
      public :
            cCalcSzWCorrel(Fonc_Num aF,Pt2di aSz,Pt2dr aSzWMax);
            void TestMultiEch_Deriche();
      private :

            Fonc_Num TrK(Fonc_Num aFonc) {return El_CTypeTraits<Type>::TronqueF(aFonc);}
            Fonc_Num MoyHigh(Fonc_Num aFonc) {return MoyIter(aFonc,2,2);}

            bool                   mIsInt;
            Pt2di                  mSz;
            Im2D<Type,TypeBase>    mImOri;
            TIm2D<Type,TypeBase>   mTImOri;
            Im2D_U_INT1            mImRes;
            Fonc_Num               mFIP;

            Im2D<Type,TypeBase>    mImEcT;
            TIm2D<Type,TypeBase>   mTImEcT;
            double                 mRatioW;
            Pt2di                  mSzW;
            Video_Win *            mW;
            Output                 mOWgr;
            Output                 mOWdisc;
            double                 mVMax;
            double                 mEcMin;
            double                 mEcMax;
            double                 mEcMoy;
            double                 mMulEc;
};


template <class Type,class TypeBase>  
cCalcSzWCorrel<Type,TypeBase>::cCalcSzWCorrel(Fonc_Num aFonc,Pt2di aSz,Pt2dr aSzWMax) :
   mIsInt     (El_CTypeTraits<Type>::IsIntType()),
   mSz        (aSz),
   mImOri     (aSz.x,aSz.y),
   mTImOri    (mImOri),
   mImRes     (aSz.x,aSz.y),
   mFIP       (mImOri.in_proj()),
   mImEcT     (aSz.x,aSz.y),
   mTImEcT    (mImEcT),
   mRatioW    (ElMin3(1.0,aSzWMax.x/mSz.x,aSzWMax.y/aSz.y)),
   mSzW       (round_ni(Pt2dr(mSz)*mRatioW)),
   mW         ((mRatioW>0) ?  Video_Win::PtrWStd(mSzW,true,Pt2dr(mRatioW,mRatioW)) : 0),
   mOWgr      (mW ? mW->ogray() : Output::onul(1)),
   mOWdisc    (mW ? mW->odisc() : Output::onul(1))
{
   ELISE_COPY
   (
       mImOri.all_pts(),
       TrK(aFonc),
       mImOri.out() | VMax(mVMax) | mOWgr
   );

   mMulEc = mIsInt ? (double(El_CTypeTraits<Type>::MaxValue()) / mVMax)   : 1.0;

  

   ELISE_COPY
   (
         mImOri.all_pts(),
         TrK(mMulEc*sqrt(Max(1e-5,Moy1(Square(mFIP))-Square(Moy1(mFIP))))),
         mImEcT.out()  |  VMax(mEcMax) | sigma(mEcMoy) |  (mOWgr << 0)
   );
   mEcMoy /= mSz.x * mSz.y;

   if (mW)
   {
       ELISE_COPY(mImEcT.all_pts(),255*mImEcT.in()/mEcMax,mOWgr);
   }

   Fonc_Num anEcFl = MoyByIterSquare(mImEcT.in_proj(),300.0,3);


   anEcFl = Max(anEcFl,mEcMoy/3);

   ELISE_COPY
   (
         mImOri.all_pts(),
         Min(255,anEcFl*30),
         mOWgr
   );

   ELISE_COPY
   (
         mImOri.all_pts(),
         mImEcT.in() < anEcFl/2,
         mImRes.out() | mOWdisc
   );

   

  Fonc_Num aFHom = MoyHigh(Square(mFIP))-Square(MoyHigh(mFIP)) <  Square(anEcFl/4.0);

   ELISE_COPY
   (
         select(mImOri.all_pts(), aFHom &&  mImRes.in() ),
         P8COL::red,
         mOWdisc | mImRes.out()
   );
   
   if (mW)
   {
        Tiff_Im::Create8BFromFonc("ImLabel.tif",mSz,mImRes.in());
   }
}




template <class Type,class TypeBase>  void cCalcSzWCorrel<Type,TypeBase>::TestMultiEch_Deriche()
{
   Im2D_REAL4 aGMax(mSz.x,mSz.y,-1);
   Im2D_INT4 aKMax(mSz.x,mSz.y,-1);

   std::vector<Im2D_REAL4> aVG;
   for (int aK=0 ; aK< 8 ; aK++)
   {
       Im2D_REAL4 aG(mSz.x,mSz.y);
       double anAlpha = 1 / (1.0+aK);
       Symb_FNum  aSF = deriche(mImOri.in_proj(),anAlpha,150);
       ELISE_COPY
       (
            mImOri.all_pts(),
            sqrt(Square(aSF.v0()) + Square(aSF.v1())),
            aG.out()
       );
       double aSom;
       ELISE_COPY(aG.all_pts(),aG.in(),sigma(aSom));
       aSom /= mSz.x * mSz.y;
       ELISE_COPY(aG.all_pts(),aG.in()/aSom,aG.out());
       if (mW)
       {
          ELISE_COPY(mImOri.all_pts(),Min(255,128*pow(aG.in(),0.5)),mW->ogray());
       }

       Fonc_Num aFK =  aK;//  Min(255,round_ni((1/anAlpha -0.5) ));
       ELISE_COPY(select(aG.all_pts(),aG.in()>aGMax.in()),Virgule(aG.in(),aFK),Virgule(aGMax.out(),aKMax.out()));
       std::cout << "AAAAAAaaaa   " << aK << "\n" ;  
   }
   if (mW)
   {
      ELISE_COPY(mImOri.all_pts(),aKMax.in(),mW->ogray());
   }
   Tiff_Im::Create8BFromFonc("KMaxDeriche.tif",mSz,aKMax.in());
   getchar();
}


template class cCalcSzWCorrel<U_INT2,INT>;
template class cCalcSzWCorrel<REAL4,REAL8>;

void Test_CalcSzWCor(int argc,char ** argv)
{
   std::string aNameIm;
   Pt2di aP0(0,0),aSz;
   // double aBigSig = 100.0;

   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(aNameIm,"Name Im"),
        LArgMain()  << EAM(aP0,"P0",true,"")
                    << EAM(aSz,"Sz",true,"")
   );

   Tiff_Im aTF = Tiff_Im::StdConvGen(aNameIm,1,false);
   if (! EAMIsInit(&aSz))
   {
      aSz = aTF.sz();
   }
   // cCalcSzWCorrel<U_INT2,INT> aCalc(trans(aTF.in_proj(),aP0),aSz,Pt2dr(900,900));
   cCalcSzWCorrel<U_INT1,INT> aCalc(trans(aTF.in_proj(),aP0),aSz,Pt2dr(900,900));

  // aCalc.TestMultiEch_Deriche();
   std::cout << "AAffffJJJ\n"; getchar();
}


void TestGrad_Nouv_0(int argc,char ** argv)
{
   std::string aNameIm;
   Pt2di aP0(0,0),aSz;
   double aBigSig = 100.0;

   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(aNameIm,"Name Im"),
        LArgMain()  << EAM(aP0,"P0",true,"")
                    << EAM(aSz,"Sz",true,"")
   );

   Tiff_Im aTF = Tiff_Im::StdConvGen(aNameIm,1,false);
   if (! EAMIsInit(&aSz))
   {
      aSz = aTF.sz();
   }
   Video_Win  aW = Video_Win::WStd(aSz,1.0);

   Im2D_REAL4 anImOri(aSz.x,aSz.y);
   ELISE_COPY
   (
        anImOri.all_pts(),
        trans(aTF.in(0),aP0),
        anImOri.out() | aW.ogray()
   );

std::cout << "JJJjjjjjjjjjjjjjjjjjj\n"; getchar();
   Im2D_REAL4 aGrad(aSz.x,aSz.y,-1);
   Im2D_REAL4 aGMax(aSz.x,aSz.y,-1);
   Im2D_INT4 aKMax(aSz.x,aSz.y,-1);

   ELISE_COPY(anImOri.all_pts(), GradBasik(anImOri.in(0)), aGrad.out());
   FilterGauss(aGrad,aBigSig);

   int aK=0;
   for (int aSzV=1 ; aSzV <= 16 ; aSzV *=2)
   {
       Im2D_REAL4 anImFlou(aSz.x,aSz.y);
       ELISE_COPY(anImOri.all_pts(),anImOri.in(),anImFlou.out());
       if (aSzV>1)
       {
          FilterGauss(anImFlou,aSzV*0.5);
       }
       Im2D_REAL4 anImEcart(aSz.x,aSz.y);
       ELISE_COPY(aW.all_pts(),EcartType(anImFlou.in_proj(),aSzV),anImEcart.out());


       ELISE_COPY(aW.all_pts(),Min(255,(anImEcart.in()/aGrad.in())*10),aW.ogray());
       Tiff_Im::Create8BFromFonc("Scale"+ToString(aSzV)+ ".tif",aSz,(anImEcart.in()/aGrad.in())*10);
       ELISE_COPY
       (
            select(anImEcart.all_pts(),anImEcart.in()>aGMax.in()),
            Virgule(anImEcart.in(),aK),
            Virgule(aGMax.out(),aKMax.out())
       );
       aK++;
   }

   Tiff_Im::Create8BFromFonc("KMax.tif",aSz,aKMax.in());
}







int MPDtest_main (int argc,char** argv)
{
    // TestGrad_Nouv_0(argc,argv);
   // TestMultiEch_Deriche(argc,argv);
    Test_CalcSzWCor(argc,argv);
//    TestKL();
//    BanniereMM3D();
   // AutoCorrel(argv[1]);
   double aNan = strtod("NAN(teta01)", NULL);
   std::cout << "Nan=" << aNan << "\n";

    return EXIT_SUCCESS;
}

#endif

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
