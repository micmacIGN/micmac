/*eLiSe06/05/99
  
     Copyright (C) 1999 Marc PIERROT DESEILLIGNY

   eLiSe : Elements of a Linux Image Software Environment

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  Author: Marc PIERROT DESEILLIGNY    IGN/MATIS  
Internet: Marc.Pierrot-Deseilligny@ign.fr
   Phone: (33) 01 43 98 81 28
eLiSe06/05/99*/


/*
   bouton 2 :  geometrie (translation/homotetie)

   bouton 3 :  menu contextuel
        
       o  recal : clik (b1/b3) sur rouge puis bleu 
       pour fixer valeur du decalage global;

*/

#include "general/all.h"
#include "private/all.h"


#include "im_tpl/image.h"
#include "MICMAC.h"
using namespace NS_ParamMICMAC;


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
            const cAppliMICMAC &     anAppli,
            VideoWin_Visu_ElImScr &aVisu,
            const std::string &   Name1,
            const std::string &   NameShrt1,
	    cElHomographie        aH1,
            const std::string &   Name2,
            const std::string &   NameShrt2,
            const std::string &   NameI2GeomInit,
	    cElHomographie        aH2
      )  : 
         BiScroller
         (
             aVisu,
             ElImScroller::StdPyramide(aVisu,Name1),
             ElImScroller::StdPyramide(aVisu,Name2,0,true,true),
             FusRG1B2,
             0,0
         ),
         mAppli    (anAppli),
         mNameXML  (      anAppli.FullDirGeom()
                       +  anAppli.PDV1()->NamePackHom(anAppli.PDV2())
/*
		      +    StdPrefixGen(NameShrt1)
                      +    std::string("_")
		      +    StdPrefixGen(NameShrt2)
		      +    "_Liaison.xml"
*/
		    ),
	 mName1     (NameShrt1),
	 mName2     (NameShrt2),
         mNameFull1 (Name1),
         mNameFull2 (Name2),
         mNameI2GeomInit (anAppli.DirImagesInit()+NameI2GeomInit),
         mH1        (aH1),
         mH2        (aH2),
         mH1_Inv    (aH1.Inverse()),
         mH2_Inv    (aH2.Inverse())
     {
          if (ELISE_fp::exist_file(mNameXML))
          {
              cElXMLTree aTree (mNameXML);
              mLiaisons = aTree.GetPackHomologues("ListeCpleHom");
              mLiaisons.ApplyHomographies(mH1_Inv,mH2_Inv);
          }
     }

     void ShowLiaison();
     void ShowLiaison(ElCplePtsHomologues &);


     void LoadXImage(Pt2di p0W,Pt2di p1W,bool quick)
     {
          BiScroller::LoadXImage(p0W,p1W,quick);
          ShowLiaison();
     }

     void write_liaison() ;

     void SauvGrille();

     ElPackHomologue   mLiaisons;
     const cAppliMICMAC & mAppli;
     std::string       mNameXML;
     std::string       mName1;
     std::string       mName2;
     std::string       mNameFull1;
     std::string       mNameFull2;
     std::string       mNameI2GeomInit;

     cElHomographie    mH1;
     cElHomographie    mH2;
     cElHomographie    mH1_Inv;
     cElHomographie    mH2_Inv;
};


void MyBisCr::SauvGrille()
{
    write_liaison() ;
    cGeomImage * aGeom = cGeomImage::Geom_DHD_Px
                         (
                              mAppli,
                              mAppli.PDV2()->SzIm(),
                              mAppli.PDV1()->ReadGridDist(),
                              mAppli.PDV2()->ReadGridDist(),
                              mAppli.PDV1()->ReadPackHom(mAppli.PDV2())
                         );

    std::string aName = ExpendPattern
                        (
                            mAppli.SL_Name_Grid_Exp().Val(),
                            mAppli.PDV1(),
                            mAppli.PDV2()
                        );
    cDbleGrid aDG
              (
                   Pt2dr(0,0),Pt2dr(SzU()),
                   mAppli.SL_Step_Grid().Val(),
                   *aGeom,
                   aName
              );
    {
       cElXMLFileIn aFileXML
                 ( 
                      mAppli.FullDirGeom()
                    + aName
                    + ".xml"
                 );
       aDG.PutXMWithData(aFileXML,mAppli.FullDirGeom());
    }
    delete aGeom;


    if (0)
    {
        cDbleGrid::cXMLMode aXmlMode;
        cDbleGrid aGr(aXmlMode,mAppli.FullDirGeom(),aName+".xml");
        while (1)
        {
            Pt2dr aP;
            cin >> aP.x >>  aP.y;
            cout << "P= " << aP << "\n";
            Pt2dr aQ = aGr.Direct(aP);
            cout << "Q= " << aQ << "\n";
            Pt2dr aR = aGr.Direct(aQ);
            cout << "R= " << aR << "\n";
        }
    }

    if (1) // Image en geometrie 1
    {
        Im2D_U_INT1 aI1 = Im2D_U_INT1::FromFile(mNameFull1);
        Pt2di aSz = aI1.sz();

        Im2D_U_INT1 aI2Init = Im2D_U_INT1::FromFile(mNameI2GeomInit);
        Im2D_U_INT1 aI2(aSz.x,aSz.y);

        cDbleGrid::cXMLMode aXmlMode;
        cDbleGrid aGr(aXmlMode,mAppli.FullDirGeom(),aName+".xml");

        Pt2di aP;
        TIm2D<U_INT1,INT> aTI2(aI2);
        TIm2D<U_INT1,INT> aTI2Init(aI2Init);

        for ( aP.x=0 ; aP.x <aSz.x; aP.x++)
           for (aP.y=0 ; aP.y <aSz.y; aP.y++)
           {
               aTI2.oset(aP,aTI2Init.get(aGr.Direct(aP),0));
           }
         Tiff_Im::Create8BFromFonc
         (
             mAppli.FullDirGeom()+"TestSup.tif",
             aSz,
             Virgule(aI1.in(),aI2.in(),aI2.in())
         );
         cout << "Done Superp\n";
    }

    if (0) // Image en geometrie 2
    {
        Im2D_U_INT1 aI1 = Im2D_U_INT1::FromFile(mNameFull1);

        Im2D_U_INT1 aI2Init = Im2D_U_INT1::FromFile(mNameI2GeomInit);
        double aFact = 5.0/3.0;
        Pt2di aSz = Pt2dr(aI1.sz()) * aFact;
        Im2D_U_INT1 aI2(aSz.x,aSz.y);

        cDbleGrid::cXMLMode aXmlMode;
        cDbleGrid aGr(aXmlMode,mAppli.FullDirGeom(),aName+".xml");

        Pt2di aP;
        TIm2D<U_INT1,INT> aTI2(aI2);
        TIm2D<U_INT1,INT> aTI1(aI1);
        // ELISE_COPY(aI1.all_pts(),((FX/10)%2)*255,aI1.out());

        for ( aP.x=0 ; aP.x <aSz.x; aP.x++)
           for (aP.y=0 ; aP.y <aSz.y; aP.y++)
           {
               int aC1 = (int)aTI1.getr(aGr.Inverse(Pt2dr(aP)/aFact),0);
               aTI2.oset(aP,aC1);
           }
         Tiff_Im::Create8BFromFonc
         (
             mAppli.FullDirGeom()+"TestSup.tif",
             aSz,
             aI2.in()
         );
         cout << "Done Superp\n";
    }



}


void MyBisCr::ShowLiaison(ElCplePtsHomologues & aCple)
{
    Video_Win  aW = this->W();
    Pt2dr aPW1 = this->Scr1().to_win(aCple.P1());
    Pt2dr aPW2 = this->Scr2().to_win(aCple.P2());
                                                                                    
    if (this->Im1Act() )
    {
       aW.draw_circle_abs
       (
            aPW1,
            3.0,
            aW.pdisc()(P8COL::red)
       );
    }
                                                                                    
    if (this->Im2Act() )
    {
       aW.draw_circle_abs
       (
            aPW2,
            3.0,
            aW.pdisc()(P8COL::blue)
       );
    }
                                                                                    
    if (this->Im1Act() && this->Im2Act() )
    {
       aW.draw_seg
       (
            aPW1,
            aPW2,
            aW.pdisc()(P8COL::magenta)
       );
    }
                                                                                    

}


void MyBisCr::ShowLiaison()
{
    for 
    (
         ElPackHomologue::tIter  itCple = mLiaisons.begin();
         itCple != mLiaisons.end();
         itCple++
    )
         ShowLiaison(*itCple);
}

void  MyBisCr::write_liaison()
{
   cElXMLFileIn aXMLFile(mNameXML);
   // aXMLFile.PutString(mParam.Directory(),"Directory");
   aXMLFile.PutString(mName1,"Name1");
   aXMLFile.PutString(mName2,"Name2");
   ElPackHomologue aPackSauv = mLiaisons;
   aPackSauv.ApplyHomographies(mH1,mH2);
   aXMLFile.PutPackHom(aPackSauv);

}


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
        Video_Win *                             pWProf;            
        VideoWin_Visu_ElImScr & mVisu;
	    MyBisCr &			mScrol;
        vector<ElImScroller*>   mVScr;
        void IncrustFromBegin(Pt2di p0,Pt2di aDec);
        REAL OptimCorrelAutom(bool addLiaison,bool Optim);

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
        CaseGPUMT *             mCaseProfil;
        CaseGPUMT *             mCasePolyg;
        CaseGPUMT *             mCaseLiaisCorr;
        CaseGPUMT *             mCaseLiaisTriv;
        CaseGPUMT *             mCaseKillLiais;
        CaseGPUMT *             mCaseSauvGrid;
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
        ElPackHomologue               & mLiaisons;
        bool                            mIsEpip;

        void SetFlagIm(INT aFlag);
        void FlipImage();

        void Recal2Pt(bool BlAndW);
        Pt2dr GetPt1Image(bool * Shifted = 0);
        Pt2dr  ToWScAct(Pt2dr aP) 
        {
             return mScrol.TheScrAct().to_win(aP);
        }

        void    KillLiaison();
        void ShowVect();
       
        void GetLiasonCorr(bool Optim);
        void Profil();
        void Profil(Im2D_REAL4 aIm,Pt2dr aP0,Pt2dr aP1,INT aCoul);


};
Pt2di aSzWP(600,200);

void  TestScroller::Profil(Im2D_REAL4 aIm,Pt2dr aP0,Pt2dr aP1,INT aCoul)
{
cout << aP0 << aP1 << aIm.sz() << "\n";
   REAL aD = euclid(aP1-aP0);
   REAL aStep = ElMax(0.1,ElMax(1.0,aD)/600.0);
   INT aNb = round_ni(aD/aStep);
   TIm2D<REAL4,REAL> aTIm(aIm);

   REAL aS0 = 0;
   REAL aS1 = 0;
   REAL aS2 = 0;
   std::vector<REAL> aVVals;
   for (INT aK=0 ; aK<=aNb ; aK++)
   {
       REAL aPds = 1-(aK/REAL(aNb));
       Pt2dr aP = barry(aPds,aP0,aP1);
       REAL aV = aTIm.getr(aP);
       aVVals.push_back(aV);
       aS0 += 1;
       aS1 += aV;
       aS2 += ElSquare(aV);
   }
   aS1 /= aS0;
   aS2 = sqrt(ElMax(1e-2,aS2/aS0 - ElSquare(aS1)));

   std::vector<Pt2dr> aVPts;
   for (INT aK=0 ; aK<=aNb ; aK++)
   {
       REAL aY = aSzWP.y * (0.5 + (aVVals[aK]-aS1)/aS2 * 0.3);
       aVPts.push_back(Pt2dr(aK,aY));
   }
   for (INT aK=0 ; aK<aNb ; aK++)
   {
       pWProf->draw_seg(aVPts[aK],aVPts[aK+1],pWProf->pdisc()(aCoul));
   }
}

void  TestScroller::Profil()
{								
    if (pWProf == 0)
    {				
       pWProf = new Video_Win(W,Video_Win::eBasG,aSzWP);
    }
    pWProf->clear();

    Clik aCl = clik_press();
    Pt2dr aPW = aCl._pt;
    W.draw_circle_abs(aPW,2.0,W.pdisc()(P8COL::green));
    Pt2dr p1 = mScrol.Scr1().to_user(aPW);
    Pt2dr p2 = mScrol.Scr2().to_user(aPW);
      
    aCl = clik_press();
    Pt2dr aQW = aCl._pt;
    W.draw_circle_abs(aQW,2.0,W.pdisc()(P8COL::green));
    W.draw_seg(aPW,aQW,W.pdisc()(P8COL::yellow));
    Pt2dr q1 = mScrol.Scr1().to_user(aQW);
    Pt2dr q2 = mScrol.Scr2().to_user(aQW);

    Pt2di aP0 = round_down(Inf(Inf(p1,q1),Inf(p2,q2))) - Pt2di(3,3);
    Pt2di aP1 = round_up  (Sup(Sup(p1,q1),Sup(p2,q2))) + Pt2di(3,3);
    Tiff_Im aF1 = Tiff_Im::StdConv(mScrol.mNameFull1);
    Tiff_Im aF2 = Tiff_Im::StdConv(mScrol.mNameFull2);

    Pt2di aSzCr = aP1-aP0;

    Im2D_REAL4 Im1(aSzCr.x,aSzCr.y);
    Im2D_REAL4 Im2(aSzCr.x,aSzCr.y);

    p1 -= aP0;
    p2 -= aP0;
    q1 -= aP0;
    q2 -= aP0;
    ELISE_COPY(Im1.all_pts(),trans(aF1.in(),aP0),Im1.out());
    ELISE_COPY(Im2.all_pts(),trans(aF2.in(),aP0),Im2.out());


    Profil(Im1,p1,q1,P8COL::blue);
    Profil(Im2,p2,q2,P8COL::red);
     

    cout <<  p1 << p2 << "\n";
    cout <<  q1 << q2 << "\n";
}



void TestScroller::ShowVect()
{
    mScrol.ShowLiaison();
    // mLiaisons.Show(mScrol);

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
   mLiaisons.RemoveNearest(aPt,true);
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

REAL TestScroller::OptimCorrelAutom(bool addLiaison,bool Optim)
{
     mLpt = mW.GetPolyg(mW.pdisc()(P8COL::red),3);
     if (Optim)
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
       mLiaisons.add(ElCplePtsHomologues(aP0,aP1,aCor));
     }

     return aCor;
}


void TestScroller::GetLiasonCorr(bool Optim)
{
         SetFlagIm(3);
         REAL aCor = OptimCorrelAutom(true,Optim);
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
    pWProf            (0),
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
     mCaseLiaisCorr  (0),
     mCaseLiaisTriv  (0),
     mCaseKillLiais  (0),
     mCaseSauvGrid   (0),
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

    mCaseSauvGrid = new CaseGPUMT
                    (
                       mPopUp,"0 0",Pt2di(3,1),
                       Tiff_Im("data/Gr.tif").in(0) *255
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


     mCaseLiaisCorr = new CaseGPUMT
                      (
                         mPopUp,"0 0",Pt2di(2,0),
                         Tiff_Im("data/PCor.tif").in(0) *255
                      );

     mCaseLiaisTriv = new CaseGPUMT
                      (
                         mPopUp,"0 0",Pt2di(3,0),
                         Tiff_Im("data/Liais.tif").in(0) *255
                      );


     mCaseFlip =      new CaseGPUMT
                      (
                         mPopUp,"0 0",Pt2di(2,1),
                         Tiff_Im("data/Flip.tif").in(0) *255
                      );

     mCaseProfil =    new CaseGPUMT
                      (
                         mPopUp,"0 0",Pt2di(2,2),
                         Tiff_Im("data/Profil.tif").in(0) *255
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
                  OptimCorrelAutom(false,true);
               if (aCase == mCaseKillLiais)
                  KillLiaison();

                if (aCase== mCaseLiaisCorr)
                  GetLiasonCorr(true); 

                if (aCase== mCaseLiaisTriv)
                  GetLiasonCorr(false); 

                if (aCase== mCaseFlip)
                   FlipImage(); 

                if (aCase== mCaseProfil)
                   Profil(); 
                if (aCase== mCaseSauvGrid)
                   mScrol.SauvGrille() ; 

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

Video_Win * BUGW=0;

void TestHomographie(const cElHomographie & aH,const std::string & aMes )
{
      Pt2dr aP0(1000,1000); Pt2dr aQ0 = aH.Direct(aP0);
      Pt2dr aP1(3000,3000); Pt2dr aQ1 = aH.Direct(aP1);
      REAL aDP = euclid(aP0,aP1);
      REAL aDQ = euclid(aQ0,aQ1);
      cout << aMes << " ; HOMOGRAPHIE : " << aDP << " -> " << aDQ << "\n";

}



bool ToImEqual
     (
         std::string & aNameIm,
         cAppliMICMAC   & anAppli
     )
{
    int aSzW = anAppli.SL_EqualIm().Val();
    if (aSzW <= 0)
       return false;

    cout << "Begin Equal " << aNameIm << "\n";
    std::string aNameInit = anAppli.DirImagesInit() 
                           + aNameIm;
    aNameIm  =    anAppli.TmpGeom().Val()
                + StdPrefixGen(aNameIm) 
                + std::string("_Equal.tif");

    std::string aFullName = anAppli.WorkDir()+aNameIm;
    if (ELISE_fp::exist_file(aFullName))
       return true;


    Tiff_Im aFileInit = Tiff_Im::StdConv(aNameInit);
    Pt2di aSz = aFileInit.sz();

    Tiff_Im aFileOut
            (
               aFullName.c_str(),
               aSz,
               GenIm::u_int1,
               Tiff_Im::No_Compr,
               Tiff_Im::BlackIsZero
            );

    Symb_FNum  Fonc (aFileInit.in(0));
    Symb_FNum  Pond (aFileInit.inside());

    Fonc_Num fSom = Virgule(Rconv(Pond),Fonc,ElSquare(Fonc));
    for (INT k=0; k< 1 ; k++)
        fSom = rect_som(fSom,aSzW)/ElSquare(1.0+2.0*aSzW);  // Pour Eviter les divergences
    Symb_FNum  S012 (fSom);

    Symb_FNum s0 (Rconv(S012.v0()));
    Symb_FNum s1 (S012.v1()/s0);
    Symb_FNum s2 (S012.v2()/s0-Square(s1));
    Symb_FNum ect  (sqrt(Max(5,s2)));
    Fonc_Num fRes = 255*erfcc((aFileInit.in()-s1)/(ect*sqrt(2.0)));

    ELISE_COPY(aFileOut.all_pts(),fRes,aFileOut.out());
    cout << "END Equal " << aNameIm << "\n";
    return true;
}


cElHomographie  ToImRedr
                (
                    bool          isNameDejaModifie,
                    std::string & aNameIm,
                    cAppliMICMAC   & anAppli,
                    ElPackHomologue  aPack,
                    std::string & aNameImBase
                )
{

    std::string aNameInit = anAppli.DirImagesInit() + aNameIm;
    Tiff_Im aTifInit(aNameInit.c_str());
    Pt2di aSz = aTifInit.sz();



    cElHomographie  aHom(aPack,true);
    // aHom = aHom.Inverse();
    Pt2di aSzOut = aSz;
    // ======

    aNameIm  =    (isNameDejaModifie ?"": anAppli.TmpGeom().Val())
                + StdPrefixGen(aNameIm) 
                + std::string("_RedrOn_")
                + StdPrefixGen(aNameImBase)
                + std::string(".tif");

    if (ELISE_fp::exist_file(anAppli.WorkDir() +aNameIm))
       return aHom;


    Im2D<U_INT1,INT> aIn = LoadFileIm(aTifInit,(U_INT1 *)0);
    Im2D<U_INT1,INT> aOut (aSzOut.x,aSzOut.y);  // Taille a changer aussi

    // U_INT1 **  DIn  = aIn.data();
    U_INT1 **  DOut = aOut.data();

    Box2di aBoxIn (Pt2di(0,0),aSz-Pt2di(1,1));
    TIm2D<U_INT1,INT> aTIn(aIn);
    cCubicInterpKernel aKer(-0.5);
    for (INT x=0 ; x<aSzOut.x ; x++)
    {
        if ((x % 10) == 0 ) 
            cout << "HOM , " << x << "\n";
        for (INT y=0 ; y<aSzOut.y ; y++)
	{
		Pt2di POut(x,y);
		Pt2dr PIn = aHom.Direct(Pt2dr(POut));
                INT aV = round_ni(aTIn.getr(aKer,PIn,0));
                DOut[POut.y][POut.x] = ElMax(0,ElMin(255,aV));
	}
    }

    Tiff_Im::CreateFromIm(aOut,anAppli.WorkDir()+aNameIm);
    return aHom;
}


int main(int argc,char** argv)
{
    Pt2di SzW(600,500);

    cAppliMICMAC & anAppli = *(cAppliMICMAC::Alloc(argc,argv,eAllocAM_Saisie));


    std::string Nameim1 = anAppli.PDV1()->Name();
    std::string Nameim2 = anAppli.PDV2()->Name();

    std::string  NameImOri1 = Nameim1;
    std::string  NameImOri2 = Nameim2;

    ToImEqual (Nameim1,anAppli);
    bool aModifName2 = ToImEqual (Nameim2,anAppli);

    bool WithEpip = anAppli.SL_Epip().Val();


    cElHomographie  Hom1 =  cElHomographie::Id();

    cElHomographie  Hom2  =  cElHomographie::Id();
    std::string  aNameH = anAppli.SL_PackHom0().Val();
    std::string aNameIm2GeomInit = Nameim2;
    if (aNameH != "")
    {
        
        cElXMLTree aTree
                   (
                          anAppli.FullDirGeom()
                       +  aNameH
                   );
        ElPackHomologue aPackExtIm2 = aTree.GetPackHomologues("ListeCpleHom");
        Hom2 =  ToImRedr(aModifName2,Nameim2,anAppli,aPackExtIm2,NameImOri1);
    }
cout << aNameIm2GeomInit << "     " << Nameim2 << "\n";

    TestHomographie(Hom1,"H1");
    TestHomographie(Hom2,"H2");






    Gray_Pal                       Pgray(90);
    Disc_Pal                       Pdisc(Disc_Pal::P8COL());
    RGB_Pal                        Prgb(3,3,3);
    Circ_Pal                       Pcirc = Circ_Pal::PCIRC6(30);
    Elise_Set_Of_Palette           SOP(NewLElPal(Pgray)+Elise_Palette(Pdisc)+Elise_Palette(Prgb)+Elise_Palette(Pcirc));
    Video_Display                  Ecr((char *) NULL);
    Video_Win                      W(Ecr,SOP,Pt2di(50,50),SzW);

    Ecr.load(SOP);

std::cout << "XX : " <<  anAppli.DirImagesInit()+Nameim2 << "\n";

     VideoWin_Visu_ElImScr  V(W,W.prgb(),SzImIncr);
     MyBisCr * aScr =  new MyBisCr 
                           (
			        anAppli,
                                V, 
                                anAppli.DirImagesInit()+Nameim1,
                                NameImOri1,
                                Hom1, 
                                anAppli.DirImagesInit()+Nameim2,
                                NameImOri2,
                                aNameIm2GeomInit,
                                Hom2
                            );

    //  aScr->SetAlwaysQuick();

      // aScr->SetTranslate(aCor0.DecEch1());



     ELISE_ASSERT(aScr !=0 , "Cannot Create Scroller \n");
     //V.SetUseEtalDyn(true);

     Pt2di aDec = WithEpip ?  
                  Pt2di(0,anAppli.SL_YDecEpip().Val()) : 
                  Pt2di(0,0);

     TestScroller(W,V,*aScr,aDec,WithEpip);



}


/********************************************************************/
/********************************************************************/
/********************************************************************/
/********************************************************************/


