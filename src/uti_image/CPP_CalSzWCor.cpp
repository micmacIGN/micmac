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







Fonc_Num Moy(Fonc_Num aF,int aSz)
{
   return rect_som(aF,aSz) / ElSquare(1.0+2*aSz);
}

Fonc_Num Moy1(Fonc_Num aF) {return Moy(aF,1);}


Fonc_Num MoyIter(Fonc_Num aF,int aSz,int aNbIter)
{
   for (int aK=0; aK<aNbIter ; aK++)
       aF = Moy(aF,aSz);

   return aF;
}



class cGradMaxLoc  : public Simple_OPBuf1<double,double>
{
   public :

      cGradMaxLoc
      (
             int     aNbV,
             double  aSeuilG
      )   :
          mNbV        (aNbV),
          mSeuilG2    (ElSquare(aSeuilG)),
          mDir        (NEW_MATRICE(Pt2di(-mNbV,-mNbV),Pt2di(mNbV+1,mNbV+1),Pt2dr))

      {
           for (int aDx=-mNbV ; aDx<=mNbV ; aDx++)
           {
              for (int aDy=-mNbV ; aDy<=mNbV ; aDy++)
              {
                   mDir[aDy][aDx] = Pt2dr(aDx,aDy);
                   if (aDx || aDy)
                      mDir[aDy][aDx] = vunit(mDir[aDy][aDx]);
              }
           }
      }

      ~cGradMaxLoc()
      {
          DELETE_MATRICE(mDir,Pt2di(-mNbV,-mNbV),Pt2di(mNbV+1,mNbV+1));
      }


   private :

      virtual void  calc_buf(double ** output,double *** input);

      int     mNbV;
      double  mSeuilG2;
      Pt2dr  **  mDir;
};

void  cGradMaxLoc::calc_buf(double ** output,double *** input)
{
    ELISE_ASSERT(this->dim_in()==2,"Incoherence in cGradMaxLoc::calc_buf");

    REAL** tgx = input[0];
    REAL** tgy = input[1];
    REAL*  res = output[0];

    for (INT x=x0() ;  x<x1() ; x++)
    {
       bool Ok=1;
       REAL gx = tgx[0][x];
       REAL gy = tgy[0][x];
       REAL g2 = ElSquare(gx)+ElSquare(gy);

       if (g2> mSeuilG2)
       {
            Pt2dr aGrad = vunit(Pt2dr(gx,gy));
            for (int aDx=-mNbV ; Ok &&(aDx<=mNbV) ; aDx++)
            {
               for (int aDy=-mNbV ; Ok &&(aDy<=mNbV) ; aDy++)
               {
                   if (
                            (g2<  ElSquare( tgx[aDy][x+aDx])+ElSquare(tgy[aDy][x+aDx]))
                        &&  ( ElAbs(scal(aGrad,mDir[aDy][aDx])) > 0.7)
                      )
                   {
                       Ok=0;
                   }
               }
            }
       }
       else
       {
          Ok= 0;
       }

       res[x] = Ok;
   }
}

Fonc_Num GradMaxLoc(Fonc_Num f,int aNbV,double aSeuilG)
{
    return create_op_buf_simple_tpl
           (
                0,
                new   cGradMaxLoc(aNbV,aSeuilG),
                f,
                1,
                Box2di ( Pt2di(-aNbV,-aNbV), Pt2di(aNbV,aNbV))
           );
}






template <class Type,class TypeBase> class cCalcSzWCorrel
{
      public :
            cCalcSzWCorrel(const std::string & aNameOut,Fonc_Num aF,Pt2di aSz,Pt2dr aSzWMax);
            void TestMultiEch_Deriche();
      private :

            Fonc_Num TrK(Fonc_Num aFonc) {return El_CTypeTraits<Type>::TronqueF(aFonc);}
            Fonc_Num MoyHigh(Fonc_Num aFonc) {return MoyIter(aFonc,2,1);}

            bool                   mIsInt;
            std::string            mNameOut;
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
            double                 mVMoy;
            double                 mV2;
            double                 mEcMin;
            double                 mEcMax;
            double                 mEcMoy;
};


template <class Type,class TypeBase>
cCalcSzWCorrel<Type,TypeBase>::cCalcSzWCorrel(const std::string & aNameOut,Fonc_Num aFonc,Pt2di aSz,Pt2dr aSzWMax) :
   mIsInt     (El_CTypeTraits<Type>::IsIntType()),
   mNameOut   (aNameOut),
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
   double aDivEcgGlob = 50;
   double aDivEcgLoc = 2;
   double aRatioGSurEc = 0.7;
   double anAlpha=1;

   double aVMax;
   ELISE_COPY(mImOri.all_pts(), aFonc, VMax(aVMax));

   double aMul  =  mIsInt ? (double(El_CTypeTraits<Type>::MaxValue()) / aVMax)   : 1.0;
   double aMulV = mIsInt ? 255.0/El_CTypeTraits<Type>::MaxValue() : 255.0/aVMax;

   Symb_FNum aFIn = Rconv(TrK(aFonc * aMul));
   ELISE_COPY
   (
       mImOri.all_pts(),
       Virgule(aFIn,Square(aFIn)),
       Virgule
       (
           mImOri.out() |  (mOWgr<< (aFIn* aMulV)) | sigma(mVMoy) ,
           sigma(mV2)
       )
   );


   mVMoy /=  mSz.x * mSz.y;
   mV2   /=  mSz.x * mSz.y;
   mV2 -= ElSquare(mVMoy);
   double anEct =  sqrt(mV2);



   ELISE_COPY
   (
         mImOri.all_pts(),
         TrK(sqrt(Max(1e-5,Moy1(Square(mFIP))-Square(Moy1(mFIP))))),
         mImEcT.out()  |  VMax(mEcMax) | sigma(mEcMoy) |  (mOWgr << 0)
   );
   mEcMoy /= mSz.x * mSz.y;
   std::cout << "ECART TYPE " << anEct << " ECART MOY " << mEcMoy << "\n";


   if (1) // ETALONNAGE DU RATIO GRAD de Deriche  / Ect
   {
       Symb_FNum  aSF = deriche(mImOri.in_proj(),anAlpha);
       double aMoyG;
       ELISE_COPY
       (
            mImOri.all_pts(),
            sqrt(Square(aSF.v0()) + Square(aSF.v1())),
            sigma(aMoyG)
       );
       aMoyG /= mSz.x * mSz.y;
       std::cout << "GradDer " << aMoyG << " " << aMoyG/mEcMoy << "\n";
       // RATIO +ou- 0.7   Gra = 0.7 mEcMoy
       // Seuil G =
   }


   if (mW)
   {
       ELISE_COPY(mImEcT.all_pts(),(255.0*sqrt(mImEcT.in()/mEcMax)),mOWgr);
       std::cout << "ECART DONE\n";
       getchar();
   }

   Fonc_Num anEcFl = MoyByIterSquare(mImEcT.in_proj(),200.0,3);


   double aSeuilHaut = anEct/ aDivEcgGlob;
   double aSeuilBas = mEcMoy/aDivEcgLoc;
   double aSeuilGrad = (aSeuilBas * aRatioGSurEc) /2.0;
   set_min_max(aSeuilBas,aSeuilHaut);
   anEcFl = Max(anEcFl,Max(aSeuilBas,aSeuilHaut));

//  std::cout << "SEUILSSS " <<  aSeuilBas << " " << aSeuilHaut  << "Seuil Grad " << aSeuilGrad << "\n";

   if (mW)
   {
      ELISE_COPY
      (
         mImOri.all_pts(),
         Min(255,(anEcFl*30)),
         mOWgr
      );
   }

   ELISE_COPY
   (
         mImOri.all_pts(),
         (mImEcT.in() < anEcFl/2.0) && (! GradMaxLoc(deriche(mImOri.in_proj(),anAlpha),3,aSeuilGrad)),
         mImRes.out() | mOWdisc
   );



  Fonc_Num aFH = mFIP;

  Fonc_Num aFHom = MoyHigh(Square(aFH))-Square(MoyHigh(aFH)) <  Square(anEcFl/2.0);

   ELISE_COPY
   (
         select(mImOri.all_pts(), aFHom &&  erod_d8(mImRes.in_proj(),1) ),
         2,
         mOWdisc | mImRes.out()
   );



   if (mW)
   {
        getchar();
   }
   Tiff_Im::Create8BFromFonc(mNameOut,mSz,mImRes.in());
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
   }
   if (mW)
   {
      ELISE_COPY(mImOri.all_pts(),aKMax.in(),mW->ogray());
   }
   Tiff_Im::Create8BFromFonc("KMaxDeriche.tif",mSz,aKMax.in());
   if (mW)
      getchar();
}


template class cCalcSzWCorrel<U_INT2,INT>;
template class cCalcSzWCorrel<REAL4,REAL8>;


int CalcSzWCor_main(int argc,char ** argv)
{
   std::string aNameIm,aNameOut;
   Pt2di aP0(0,0),aSz;
   Pt2dr aSzW(0,0);
   bool ForceFloat=true;
   // double aBigSig = 100.0;

   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(aNameIm,"Image name", eSAM_IsExistFile),
        LArgMain()  << EAM(aP0,"P0",true,"Origin (Def=(0,0))")
                    << EAM(aSz,"Sz",true,"Size (Def full size of tiff file)", eSAM_NoInit)
                    << EAM(aSzW,"Wsz",true,"Size of window (Def no window)")
                    << EAM(aNameOut,"Out",true,"Out")
                    // << EAM(ForceFloat,"FF",true,"Force Float Tmp Image (tuning purpose)")  !! BUGUE EN INT2 ????
   );

   if (!MMVisualMode)
   {
       Tiff_Im aTF = Tiff_Im::StdConvGen(aNameIm,1,true);
       if (! EAMIsInit(&aSz))
       {
          aSz = aTF.sz() -aP0;
       }

       if (!EAMIsInit(&aNameOut))
       {
           std::string aDir,aName;
           SplitDirAndFile(aDir,aName,aNameIm);
           aNameOut = aDir + "ImSzW_" + aName + ".tif";
       }

       // cCalcSzWCorrel<U_INT2,INT> aCalc(trans(aTF.in_proj(),aP0),aSz,Pt2dr(900,900));

       if (
               ((aTF.type_el()==GenIm::u_int1) || (aTF.type_el()==GenIm::u_int2))
            && (! ForceFloat)
          )
       {
          cCalcSzWCorrel<U_INT2,INT> aCalc(aNameOut,trans(aTF.in_proj(),aP0),aSz,aSzW);
       }
       else
       {
          cCalcSzWCorrel<REAL4,REAL8> aCalc(aNameOut,trans(aTF.in_proj(),aP0),aSz,aSzW);
       }

      // aCalc.TestMultiEch_Deriche();

       return EXIT_SUCCESS;
   }
   else
       return EXIT_SUCCESS;
}


//==================================================================
//==================================================================
//==================================================================
//==================================================================


#if (0)


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

   Tiff_Im aTF = Tiff_Im::StdConvGen(aNameIm,1,true);
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





#endif

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
