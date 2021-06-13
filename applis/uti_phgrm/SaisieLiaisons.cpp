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


/*
   bouton 2 :  geometrie (translation/homotetie)

   bouton 3 :  menu contextuel
        
       o  recal : clik (b1/b3) sur rouge puis bleu 
       pour fixer valeur du decalage global;

*/


#include "../Anag/anag_all.h"
#include "../Anag/Liaisons.cpp"

Pt2di SzImIncr(300,300);

class Incruster : public ElImIncruster
{
     public :

           static const INT BrdMax = 100;
           static const INT DecMax = 40;
           static const INT StdBrd = 10;

            INT SzBrd() const
            {
                return ElMin(BrdMax,round_ni(StdBrd*mScale));
            }
            Incruster
            (
                Pt2di                           SzIm,
                Visu_ElImScr &                  Visu,
                const vector<ElImScroller*>   & Scrols
            )  :
               ElImIncruster(SzIm,Pt2di(BrdMax+DecMax,BrdMax+DecMax),Visu,Scrols,3),
               mDec12 (0,0)
            {
            }

            void SetDec12(Pt2di aDec) 
            {
                 mDec12 = Inf(Pt2di(DecMax,DecMax),Sup(-Pt2di(DecMax,DecMax),aDec));

            }
            Pt2di  CurDec() const {return mDec12;}

            INT Filters
                (
                   ElSTDNS vector<ElImIncr_ScrAttr> & Attrs,
                   ElSTDNS vector<Im2D_INT4> & ImsOut,
                   ElSTDNS vector<Im2D_INT4> & ImsIn,
                   Pt2di p0,Pt2di p1
                );

            bool PixelIndep() const
            {
                 return false;
            }

            void SetScale(REAL aScale)
            {
                 mScale = aScale;
            }

        private :

             Filtr_Incr_EqDyn_Glob mFIEG;
             Filtr_Incr_EqDyn_Loc  mFIEL;
             REAL                  mScale;
             Pt2di                 mDec12;



};

Fonc_Num AUC(Fonc_Num f)
{
   return Max(0,Min(255,f));
}

INT Incruster::Filters
    (
       ElSTDNS vector<ElImIncr_ScrAttr> & Attrs,
       ElSTDNS vector<Im2D_INT4> & ImsOut,
       ElSTDNS vector<Im2D_INT4> & ImsIn,
       Pt2di p0,Pt2di p1
    )
{
    INT fact = 2;


    mFIEG.Filters(Attrs,ImsOut,ImsIn,p0,p1);
    // mFIEL.Filters(Attrs,ImsOut,ImsIn,p0,p1,SzBrd());

    Symb_FNum f1 = ImsOut[0].in(0);
    Symb_FNum f2 = trans(ImsOut[1].in(0),mDec12);
    Symb_FNum  dif 	 = fact * (f1-f2);

    ELISE_COPY
    (  
          rectangle(p0,p1),  
          Virgule(AUC(f1+dif), (f1+f2)/2,AUC(f2-dif)),
          Virgule(ImsOut[2].out() , ImsOut[3].out() , ImsOut[4].out())
    );
    ELISE_COPY
    (  
          rectangle(p0,p1),  
          Virgule(ImsOut[2].in() , ImsOut[3].in() , ImsOut[4].in()),
          Virgule(ImsOut[0].out() , ImsOut[1].out() , ImsOut[2].out())
    );
    return 3;
}




/***************************************************/
/***************************************************/
/***                                             ***/
/***    TestScroller                             ***/
/***                                             ***/
/***************************************************/
/***************************************************/

class MyBisCr : public BiScroller
{
    public :
      MyBisCr
      (
            VideoWin_Visu_ElImScr &aVisu,
            const std::string &   Name1,
	    cElHomographie        aH1,
            const std::string &   Name2,
	    cElHomographie        aH2,
            const std::string &   NameLiaison
      )  : 
         BiScroller
         (
             aVisu,
             ElImScroller::StdPyramide(aVisu,Name1),
             ElImScroller::StdPyramide(aVisu,Name2,0,true,true),
             FusRG1B2,
             0,0
         ),
         mLiaisons (NameLiaison,aH1,aH2),
         mNameL    (NameLiaison)
     {
     }

     void LoadXImage(Pt2di p0W,Pt2di p1W,bool quick)
     {
          BiScroller::LoadXImage(p0W,p1W,quick);
          mLiaisons.Show(*this);
     }

     void write_liaison() {/*mLiaisons.write(mNameL);*/ cout << "No W\n";}

     EnsLiaisons   mLiaisons;
     string        mNameL;
 
};

static const INT NbPixInitCorr = 2;

class TestScroller :
           public   Optim2DParam,
           private  EliseStdImageInteractor,
	   public   Grab_Untill_Realeased
{
	public :

                REAL Op2DParam_ComputeScore(REAL,REAL) ;

		virtual ~TestScroller(){};
		TestScroller
                (
                   Video_Win,
                   VideoWin_Visu_ElImScr & aVisu,
                   MyBisCr & aScr,
                   Pt2di     aTrInit,
                   bool  IsEpip
                 );

		void GUR_query_pointer(Clik p,bool);
		void GUR_button_released(Clik p);
                void OnEndTranslate(Clik){ShowVect();}
                void OnEndScale(Clik aCl)
                {
                   EliseStdImageInteractor::OnEndScale(aCl);
                   ShowVect();
                }


                Pt2dr SetTranslate(Pt2dr aP) 
                {
                      if (mIsEpip) aP.y = 0;
                      mScrol.SetTranslate(aP);
                      return aP;
                }


        Video_Win				W;            
        VideoWin_Visu_ElImScr & mVisu;
	    MyBisCr &			mScrol;
        vector<ElImScroller*>   mVScr;
        void IncrustFromBegin(Pt2di p0,Pt2di aDec);
        REAL OptimCorrelAutom(bool addLiaison);

	    Pt2di 	    			mLastCl;
	    Pt2di 	    			mLastDec;
	    Pt2di 	    			mIncrCenter;
	    bool                    mModePopUp;
	    bool                    mModeIncr;
        INT                     mCptScale;
        Tiff_Im                 mTifMenu;
        Pt2di                   mSzCase;
        GridPopUpMenuTransp     mPopUp;
        ChoixParmiCaseGPUMT *   mCase12;
        ChoixParmiCaseGPUMT *   mCase1;
        ChoixParmiCaseGPUMT *   mCase2;
        CaseGPUMT *             mCaseExit;
        CaseGPUMT *             mCaseRecal;
        CaseGPUMT *             mCaseFlip;
        CaseGPUMT *             mCasePolyg;
        CaseGPUMT *             mCaseSeg;
        CaseGPUMT *             mCaseLiaisInter;
        CaseGPUMT *             mCaseLiaisCorr;
        CaseGPUMT *             mCaseKillLiais;
        BoolCaseGPUMT *         mCaseModeGr;
        Incruster               mIncr;
        ElList<Pt2di> mLpt;
        INT                     mFlagIm;
        bool                    mBlocSwapIm;

        INT                             mMaxSeg;
        INT                             mRabSeg;
        INT                             mBoxSeg;
        Memory_ElImDest<U_INT1>         aMem;
        Im2D_INT1                       mGX;
        Im2D_INT1                       mGY;
        Im2D_U_INT1                     mRhoG;
        EnsLiaisons  & mLiaisons;
        bool                            mIsEpip;

        void SetFlagIm(INT aFlag);
        void FlipImage();

        void Recal2Pt(bool BlAndW);
        Pt2dr GetPt1Image(bool * Shifted = 0);
        Seg2d  GetSeg1Image(INT Coul = -1,bool * Shifted  = 0);
        Pt2dr  ToWScAct(Pt2dr aP) 
        {
             return mScrol.TheScrAct().to_win(aP);
        }

        Seg2d  AffineSeg(SegComp,ElImScroller &,INT Coul = -1,bool Im=false);
        void    KillLiaison();
        void ShowVect();
       
        void GetLiasonInter();
        void GetLiasonCorr();


};




void TestScroller::ShowVect()
{
    mLiaisons.Show(mScrol);

}

Pt2dr TestScroller::GetPt1Image(bool * shifted)
{
    Clik aCl = clik_press();
    if (shifted) *shifted = aCl.shifted();
    return mScrol.TheScrAct().to_user(aCl._pt);
    
}

void    TestScroller::KillLiaison()
{
   Pt2dr aPt = mScrol.TheFirstScrAct().to_user(clik_press()._pt);
   aPt = mScrol.TheFirstScrAct().to_win(aPt);
   aPt = mScrol.Scr1().to_user(aPt);
   mLiaisons.remove_nearestP0(aPt);
}


Seg2d TestScroller::GetSeg1Image(INT aCoul,bool * shifted)
{
   bool sh1,sh2;
   mBlocSwapIm = true;
   Pt2dr aP1 = GetPt1Image(&sh1);
   Pt2dr aP2 = GetPt1Image(&sh2);
   if (shifted) 
       *shifted = sh1 &sh2;
   mBlocSwapIm = false;

   if (aCoul >=0)
      W.draw_seg
      (
          ToWScAct(aP1),
          ToWScAct(aP2),
          W.pdisc()(aCoul)
      );

   return Seg2d(aP1,aP2);
}


Seg2d  TestScroller::AffineSeg(SegComp aSeg,ElImScroller & aScr,INT aCoul,bool modeIm)
{

    if (dist8(aSeg.v01()) > mMaxSeg)
       return aSeg;


    Pt2di aTr = Inf(aSeg.p0(),aSeg.p1()) - Pt2dr(mRabSeg,mRabSeg);
     
    Pt2di aSz =  round_up(Sup(aSeg.p0(),aSeg.p1()) -Inf(aSeg.p0(),aSeg.p1())) 
               + Pt2dr(mRabSeg,mRabSeg) * 2;

    aScr.LoadXImageInVisu
    (
         aMem,
         Pt2di(0,0), aSz,
         false,
         aTr, 1.0
    );

    if (! mCaseModeGr->Val())
    {
       ELISE_COPY
       (
          rectangle(Pt2di(0,0),aSz),
          aMem.Images()[0].in(),
          mRhoG.out()
       );
    }
    else
    {
       ELISE_COPY
       (
          rectangle(Pt2di(0,0),aSz),
          Max(-128,Min(127,deriche(aMem.Images()[0].in_proj(),1.7,20))),
          Virgule(mGX.out(),mGY.out())
       );

       ELISE_COPY
       (
          rectangle(Pt2di(0,0),aSz),
          sqrt(Square(mGX.in())+Square(mGY.in())),
          mRhoG.out()
       );
   }

    aSeg = aSeg.trans(-aTr);
    double aScore;
    SegComp aSegAff = OptimizeSegTournantSomIm
                    (
                        aScore,
                        mRhoG,
                        aSeg,
                        round_ni(ElMax(aSeg.length(),10.0)),
                        1.0,
                        0.01
                    );

/*
    Video_Win aW = W.chc(Pt2dr(0,0),Pt2dr(5.0,5.0));
    ELISE_COPY
    (
          rectangle(Pt2di(0,0),aSz),
          Min(255,3*mRhoG.in(0)),
          aW.ogray()
    );

    aW.draw_seg(aSeg.p0(),aSeg.p1(),aW.pdisc()(P8COL::blue));
    aW.draw_seg(aSegAff.p0(),aSegAff.p1(),aW.pdisc()(P8COL::red));
*/
   aSegAff = aSegAff.trans(aTr);
   
   if (aCoul >=0)
      W.draw_seg
      (
            ToWScAct(aSegAff.p0()),
            ToWScAct(aSegAff.p1()),
            W.pdisc()(aCoul)
      );

    return aSegAff;
}

void TestScroller::GetLiasonInter()
{
     vector<Seg2d> aVS; 
     for (INT k=0 ; k<4 ; k++)
     {
         SetFlagIm((k<2) ? 1 : 2);
         if (! (k&1))
               mScrol.LoadAndVisuIm();

         Seg2d aSeg = GetSeg1Image();
         aVS.push_back(AffineSeg(aSeg,mScrol.TheScrAct(),P8COL::red));
     }

     mLiaisons.AddInter(aVS[0],aVS[1],aVS[2],aVS[3]);
     SetFlagIm(3);
}



INT SensibRec = 3;


void TestScroller::SetFlagIm(INT aFlag)
{
cout << mFlagIm << " => " << aFlag << "\n";
    if (aFlag == mFlagIm)
       return;

    mFlagIm = aFlag;
    mScrol.SetImAct((bool)(aFlag&1),(bool)(aFlag&2));
    mScrol.LoadAndVisuIm();
    ShowVect();
}

void TestScroller::FlipImage()
{
    INT aCurFlag = mFlagIm;
    for (INT k=0 ; k<20 ; k++)
       SetFlagIm(1+(k%2));

    SetFlagIm(aCurFlag);
}



void TestScroller::GUR_button_released(Clik cl)
{
     if (cl.controled())
     {
          SetTranslate
          (
                 mScrol.CurTr() 
              +  ((mLastDec-cl._pt)/SensibRec) / mScr.sc() 
          );
          cout  << "TR= " << mScrol.Scr1().tr() - mScrol.Scr2().tr() << "\n";
     }
}

void TestScroller::GUR_query_pointer(Clik cl,bool)
{
     if (mModePopUp)
     {
        mPopUp.SetPtActif( cl._pt);
     }

     if (mModeIncr)
     {
         if (!cl.controled())
           mLastDec = cl._pt;

         if (cl.shifted())
         {
             INT dy = (INT) cl._pt.y-mLastCl.y;
             mIncr.IncrSzAndReaff(mIncrCenter,Pt2di(dy,dy),W.prgb());
         }
         else if (cl.controled())
         {
             IncrustFromBegin(mIncrCenter,(mLastDec-cl._pt)/SensibRec);
         }
         else
         {
            mIncrCenter = cl._pt;
            mIncr.IncrustCenterAtPtW(cl._pt,W.prgb());
         }
     }

     mLastCl =  cl._pt;
}

void TestScroller::Recal2Pt(bool BlackAndW)
{
cout << "BLW = " << BlackAndW << "\n";
    INT aFlag = mFlagIm;
    if (BlackAndW) 
       SetFlagIm(1);
    Clik cl1 = clik_press();
    Pt2dr p1 = mScrol.Scr1().to_user(cl1._pt);
    if (BlackAndW) 
       SetFlagIm(2);
    Clik cl2   = clik_press();
    Pt2dr p2 = mScrol.Scr2().to_user(cl2._pt);
    SetFlagIm(aFlag);
    SetTranslate(p2-p1);
    ShowVect();

    cout << "Translate = " << p2-p1 << "\n";
}



REAL TestScroller::Op2DParam_ComputeScore(REAL xR,REAL yR)
{
   Pt2di dec =  Pt2dr(xR,yR) * NbPixInitCorr;

   Symb_FNum   f1 = Rconv(mScrol.Im1().in(0));
   Symb_FNum   f2 = trans(Rconv(mScrol.Im2().in(0)),dec);

   REAL S[6];

   ELISE_COPY
   (
         polygone(mLpt),
         Virgule(1.0,f1,f2,Square(f1),f1*f2,Square(f2)),
         sigma(S,6)
   );
 
   REAL s = S[0];
   REAL s1= S[1] /s;
   REAL s2= S[2] /s;

   REAL s11 = S[3]/s - ElSquare(s1);
   REAL s12 = S[4]/s - s1 * s2;
   REAL s22 = S[5]/s - ElSquare(s2);


   REAL coeff = s12 / sqrt(ElMax(s11*s22,0.1));

cout << "Coeff = "  << coeff << "\n";
   return coeff;
}

REAL TestScroller::OptimCorrelAutom(bool addLiaison)
{
     mLpt = mW.GetPolyg(mW.pdisc()(P8COL::red),3);
     Optim2DParam::optim();


     REAL S[3];
     ELISE_COPY
     (
         polygone(mLpt),
         Virgule(1.0,FX,FY),
         sigma(S,3)
     );

     Pt2dr aDec = param() * NbPixInitCorr;

     // IncrustFromBegin(aC,aDec);

     Clik aCl= mW.clik_in();

     Pt2dr aTrans = mScrol.CurTr()+ aDec/mScr.sc();
     SetTranslate(aTrans);

     REAL aCor = Op2DParam_ComputeScore(0,0);

     if (addLiaison)
     {
       Pt2dr aC = Pt2dr(S[1],S[2])/ S[0];
       Pt2dr aP0 = mScrol.Scr1().to_user(aC);
       Pt2dr aP1 = aP0+aTrans;
       cout << "P0 Pokyg" << aP0 << aP1 << "\n";
       mLiaisons.AddCorrel(aP0,aP1,aCor);
     }

     return aCor;
}


void TestScroller::GetLiasonCorr()
{
         SetFlagIm(3);
         REAL aCor = OptimCorrelAutom(true);
         // Clik aCl = clik_press();

         cout << "Cor = " << aCor << "\n";
}


void TestScroller::IncrustFromBegin(Pt2di aP,Pt2di aDec)
{
      if (mIsEpip) 
         aDec.y = 0;
      mIncr.SetDec12(aDec);
      mIncr.SetScale(mScrol.sc());
      mIncr.BeginIncrust
      (
           aP,
           mScr.sc() 
      );
      mIncr.IncrustCenterAtPtW(aP,W.prgb());
}


TestScroller::TestScroller
(
     Video_Win WIN,
     VideoWin_Visu_ElImScr & aVisu,
     MyBisCr & SCROL,
     Pt2di aTrInit,
     bool  IsEpip
) :
   Optim2DParam ( 0.9 / NbPixInitCorr, -10, 1e-5, true ),
   EliseStdImageInteractor(WIN,SCROL,2),
   W			  (WIN),
    mVisu         (aVisu),
	mScrol         (SCROL),
    mVScr         (SCROL.SubScrolls()),
    mCptScale     (0),
    mTifMenu      ("data/Loupe.tif"),
    mSzCase       (mTifMenu.sz()),
    mPopUp        (
                      WIN,
                      mSzCase,
                      Pt2di(5,5),
                      Pt2di(1,1)
                  ),
     mCase12      (0),
     mCase1      (0),
     mCase2      (0),
     mCaseExit    (0),
     mCaseRecal   (0),
     mCasePolyg   (0),
     mCaseSeg     (0),
     mCaseLiaisInter (0),
     mCaseLiaisCorr  (0),
     mCaseKillLiais  (0),
     mCaseModeGr     (0),
     mIncr        (SzImIncr,aVisu,mVScr),
     mFlagIm      (3),
     mBlocSwapIm  (false),
     mMaxSeg      (500),
     mRabSeg      (20),
     mBoxSeg      (mMaxSeg + 2 * mRabSeg),
     aMem         (1,Pt2di(mBoxSeg,mBoxSeg)),
     mGX          (mBoxSeg,mBoxSeg),
     mGY          (mBoxSeg,mBoxSeg),
     mRhoG        (mBoxSeg,mBoxSeg),
     mLiaisons    (mScrol.mLiaisons),
     mIsEpip      (IsEpip)
{

  //  mVisu.SetEtalDyn(0,1000);


    mCase12 =
        new ChoixParmiCaseGPUMT
        (
             mPopUp,"1 0",Pt2di(1,0),
             (!Tiff_Im("data/i12.tif").in(0)) * 255,
             Tiff_Im("data/i12.tif").in(0) * 255,
             3,
             (ChoixParmiCaseGPUMT *)0
        );
    mCase1 = new ChoixParmiCaseGPUMT
        (
             mPopUp,"0 0",Pt2di(1,1),
             (!Tiff_Im("data/i1.tif").in(0)) * 255,
             Tiff_Im("data/i1.tif").in(0) * 255,
             1,
             mCase12
        );
    mCase2 =new ChoixParmiCaseGPUMT
        (
             mPopUp,"0 0",Pt2di(1,2),
             (!Tiff_Im("data/i2.tif").in(0)) * 255,
             Tiff_Im("data/i2.tif").in(0) * 255,
             2,
             mCase12
        );


    mCaseKillLiais = new CaseGPUMT
                    (
                       mPopUp,"0 0",Pt2di(1,3),
                       Tiff_Im("data/TDM.tif").in(0) *255
                    );

    
     mCaseModeGr = new BoolCaseGPUMT
        (
             mPopUp,"0 0",Pt2di(1,4),
             Tiff_Im("data/Gr.tif").in(0) * 255,
             Tiff_Im("data/Im.tif").in(0) * 255,
             true
        );



    mCaseExit = new CaseGPUMT
                    (
                       mPopUp,"0 0",Pt2di(4,4),
                       Tiff_Im("data/Exit.tif").in(0) *255
                    );

    mCaseRecal = new CaseGPUMT
                    (
                       mPopUp,"0 0",Pt2di(0,1),
                       Tiff_Im("data/Recal.tif").in(0) *255
                    );

    mCasePolyg = new CaseGPUMT
                    (
                       mPopUp,"0 0",Pt2di(0,2),
                       Tiff_Im("data/Polyg.tif").in(0) *255
                    );

    mCaseSeg = new CaseGPUMT
                    (
                       mPopUp,"0 0",Pt2di(0,3),
                       Tiff_Im("data/Dr.tif").in(0) *255
                    );

    mCaseLiaisInter = new CaseGPUMT
                    (
                       mPopUp,"0 0",Pt2di(0,4),
                       Tiff_Im("data/LiaisInter.tif").in(0) *255
                    );

     mCaseLiaisCorr = new CaseGPUMT
                      (
                         mPopUp,"0 0",Pt2di(2,0),
                         Tiff_Im("data/PCor.tif").in(0) *255
                      );

     mCaseFlip =      new CaseGPUMT
                      (
                         mPopUp,"0 0",Pt2di(2,1),
                         Tiff_Im("data/Flip.tif").in(0) *255
                      );






    // mScrol.SetTranslate(aTrInit);
    // mScrol.set(Pt2dr(0,0),0.2,false);

    
    mCptScale++;


     CaseGPUMT * aCase = 0;

     while (aCase != mCaseExit)
     {
         Clik cl1   = clik_press();
         mLastCl =  cl1._pt;
         mLastDec =  cl1._pt;
         mIncrCenter = cl1._pt;
         mModePopUp = false;
         mModeIncr = false;

         switch (cl1._b )
         {

             case 1 :
             {
                 mModeIncr = true;
                 IncrustFromBegin(cl1._pt,Pt2di(0,0));
                 W.grab(*this);
                 mIncr.EndIncrust();
                 cout << "SCALE = " << mScrol.sc() << "\n";
             }
             break;

             case 3 : 
             {
               mModePopUp = true;
               mPopUp.UpCenter(cl1._pt);
               W.grab(*this);
               aCase = mPopUp.PopAndGet();

               if (aCase == mCaseRecal)
                   Recal2Pt(cl1.shifted());
               if (aCase == mCasePolyg)
                  OptimCorrelAutom(false);
               if (aCase == mCaseKillLiais)
                  KillLiaison();
               if (aCase == mCaseSeg)
               {
                  bool sh;
                  Seg2d aSeg = GetSeg1Image(P8COL::blue,&sh);
                  AffineSeg(aSeg,mScrol.TheScrAct(),P8COL::red,sh);
                }
               if(aCase== mCaseLiaisInter)
                  GetLiasonInter(); 

                if (aCase== mCaseLiaisCorr)
                  GetLiasonCorr(); 

                if (aCase== mCaseFlip)
                   FlipImage(); 

                if (
                         (aCase == mCase12)
                     ||  (aCase == mCase1)
                     ||  (aCase == mCase2)
                   )
                   SetFlagIm(mCase12->IdSelected());
                cout << "Id 12 " << mCase12->IdSelected() << "\n";
             }
             break;
         }
    }
   
    mScrol.write_liaison();

}

std::string ToStr(int aV)
{
   char aBuf[50];
   sprintf(aBuf,"%d",aV);
 
   return std::string(aBuf);
}

Video_Win * BUGW=0;

/*
cElHomographie  ToImRedr(const std::string & aNameIdent,std::string & aNameIm,ParamAnag & aParam,INT aNum)
{
    if (! aParam.ImageA180(aNameIdent))
       return cElHomographie::Id();


    std::string aNameInit = cAnagFileInit::NameFileImage(aParam,aNum,1);
    Tiff_Im aTifInit(aNameInit.c_str());
    Pt2di aSz = aTifInit.sz();

    // Partie a changer si autre chose que 180
    ElPackHomologue aPack; 
    Pt2dr aP0(0,0); 
    Pt2dr aP1(aSz.x-1,aSz.y-1);
    aPack.add(ElCplePtsHomologues(aP0,aP1));
    aPack.add(ElCplePtsHomologues(aP1,aP0));
    cElHomographie  aHom(aPack,true);
    Pt2di aSzOut = aSz;
    // ======

    aNameIm  = cAnagFileInit::NameFileImage(aParam,aNum,1,"Redr");

    if (ELISE_fp::exist_file(aNameIm.c_str()))
       return aHom;


    Im2D<U_INT1,INT> aIn = LoadFileIm(aTifInit,(U_INT1 *)0);
    Im2D<U_INT1,INT> aOut (aSzOut.x,aSzOut.y);  // Taille a changer aussi

    U_INT1 **  DIn  = aIn.data();
    U_INT1 **  DOut = aOut.data();

    Box2di aBoxIn (Pt2di(0,0),aSz);
    for (INT x=0 ; x<aSzOut.x ; x++)
    {
        if ((x % 10) == 0 ) 
            cout << "HOM , " << x << "\n";
        for (INT y=0 ; y<aSzOut.y ; y++)
	{
		Pt2di POut(x,y);
		Pt2di PIn = aHom.Direct(Pt2dr(POut));
		if (aBoxIn.inside(PIn))
		{
                   DOut[POut.y][POut.x] = DIn[PIn.y][PIn.x];
		}
	}
    }

    Tiff_Im::CreateFromIm(aOut,aNameIm);
    return aHom;
}
*/



void Bench_Visu_ISC(INT argc,char ** argv)
{

    Pt2di SzW(600,600);

    Gray_Pal                       Pgray(90);
    Disc_Pal                       Pdisc(Disc_Pal::P8COL());
    RGB_Pal                        Prgb(3,3,3);
    Circ_Pal                       Pcirc = Circ_Pal::PCIRC6(30);
    Elise_Set_Of_Palette           SOP(NewLElPal(Pgray)+Elise_Palette(Pdisc)+Elise_Palette(Prgb)+Elise_Palette(Pcirc));
    Video_Display                  Ecr((char *) NULL);
    Video_Win                      W(Ecr,SOP,Pt2di(50,50),SzW);

    Ecr.load(SOP);

    


     VideoWin_Visu_ElImScr  V(W,W.prgb(),SzImIncr);
     /*
     MyBisCr * aScr =  new MyBisCr 
                           (
                                V, 
                                aNameScr1,
                                Hom1, 
                                aNameScr2,
                                Hom2,
                                NameLiaisons
                            );
			    */
     MyBisCr * aScr =  new MyBisCr 
                           (
                                V, 
                                "/home/pierrot/Data/AutoCalib/bleu1/CD8",
				cElHomographie::Id(), 
                                "/home/pierrot/Data/AutoCalib/bleu1/CG6",
				cElHomographie::Id(),
                                "aFile.liais"
                            );


    //  aScr->SetAlwaysQuick();

      // aScr->SetTranslate(aCor0.DecEch1());



     ELISE_ASSERT(aScr !=0 , "Cannot Create Scroller \n");
     V.SetUseEtalDyn(true);

     bool WithEpip = false;
     Pt2di aDec =  Pt2di(0,0) ;

     TestScroller(W,V,*aScr,aDec,WithEpip);
}

/********************************************************************/
/********************************************************************/
/********************************************************************/
/********************************************************************/

int main(int argc,char** argv)
{
     Bench_Visu_ISC(argc,argv);
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
