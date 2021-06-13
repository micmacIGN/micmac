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

#include "StdAfx.h"


#if (ELISE_X11)
/*************************************************/
/*                                               */
/*                XXXXXXX                        */
/*                                               */
/*************************************************/


ElImScroller * SCR(Visu_ElImScr &aVisu,const std::string & aName,bool ForceGray)
{
    Tiff_Im aTifFile = Tiff_Im::StdConvGen(aName,-1,true);

    // INT aDim = aTifFile.nb_chan(); aDim=1;

    aVisu.AdaptTiffFile(aTifFile,true,ForceGray);  // AdaptPal,ForceGray);
    switch (aTifFile.type_el())
    {
    case GenIm::u_int1 :
        return new ImFileScroller<U_INT1> (aVisu,aTifFile,1.0);
        break;
    case GenIm::u_int2 :
        return new ImFileScroller<U_INT2> (aVisu,aTifFile,1.0);
        break;

    default :
        break;
    }

    ELISE_ASSERT(false,"Uncompatible file type");

    return 0;

    // return new ImFileScroller<U_INT1> (aVisu,aTifFile,1);

    // return ElImScroller::StdScrollIfExist(aVisu,aName,1.0,true,true);
}


cWinIm::cWinIm(cAppli_SaisiePts& anAppli,Video_Win aW,Video_Win aWT,cImage & aIm0) :
    Grab_Untill_Realeased(),
    mAppli   (anAppli),
    mUseMMPt (anAppli.Param().SectionWindows().UseMinMaxPt().Val()),
    mW (aW),
    mWT (aWT),
    mVWV (aW,StdPalOfFile(aIm0.Tif().name(),aW),Pt2di(10,10)),  // Sz  Incrustation
    mScr    (SCR(mVWV,aIm0.Tif().name(),mAppli.Param().ForceGray().Val())),
    mCurIm          (0),
    mModeRelication (true),
    mSzW            (mW.sz()),
    mSzCase         (MMIcone("Exit").sz()),
    mPopUpBase      ( new GridPopUpMenuTransp(mW,mSzCase,Pt2di(2,3),Pt2di(1,1))),
    mPopUpShift     ( new GridPopUpMenuTransp(mW,mSzCase,Pt2di(2,3),Pt2di(1,1))),
    mPopUpCtrl      ( new GridPopUpMenuTransp(mW,Pt2di(50,33),Pt2di(1,4),Pt2di(1,1))),
    mPopUp1Shift    ( new GridPopUpMenuTransp(mW,Pt2di(50,33),Pt2di(4,3),Pt2di(1,1))),
    mPopUpCur       (0),

    mCaseExit       (new CaseGPUMT
                     (
                         *mPopUpShift,"titi",Pt2di(0,0),
                         MMIcone("Exit").in(1) *255
                         )
                     ),

    mCaseVide       (new CaseGPUMT
                     (
                         *mPopUpBase,"titi",Pt2di(1,1),
                         MMIcone("Vide").in(1) *255
                         )
                     ),
    mCaseTDM       (new CaseGPUMT
                    (
                        *mPopUpBase,"titi",Pt2di(1,0),
                        MMIcone("TDM").in(1) *255
                        )
                    ),

    mCaseInterrog     (new CaseGPUMT
                       (
                           *mPopUpBase,"titi",Pt2di(0,0),
                           MMIcone("interog").in(1) *255
                           )
                       ),


    mCaseSmile       (new CaseGPUMT
                      (
                          *mPopUpBase,"titi",Pt2di(0,1),
                          MMIcone("Smile;-)").in(1) *255
                          )
                      ),
    mBCaseVisiRefut  ( new BoolCaseGPUMT
                       (
                           *mPopUpShift,"tata",Pt2di(0,1),
                           MMIcone("RefVis").in(1) *255,
                           MMIcone("RefInv").in(1) *255,
                           false
                           )
                       ),
    mBCaseShowDet  ( new BoolCaseGPUMT
                     (
                         *mPopUpShift,"tata",Pt2di(0,2),
                         MMIcone("NoDet").in(1) *255,
                         MMIcone("Det").in(1) *255,
                         mAppli.ShowDet()
                         )
                     ),
    mCaseHighLight    (new CaseGPUMT
                       (
                           *mPopUpBase,"titi",Pt2di(0,2),
                           MMIcone("HL").in(1) *255
                           )
                       ),
    mCaseUndo       (new CaseGPUMT
                     (
                         *mPopUpShift,"titi",Pt2di(1,0),
                         MMIcone("Undo").in(1) *255
                         )
                     ),
    mCaseRedo       (new CaseGPUMT
                     (
                         *mPopUpShift,"titi",Pt2di(1,1),
                         MMIcone("Redo").in(1) *255
                         )
                     ),
    mCaseAllW       (new CaseGPUMT
                     (
                         *mPopUpCtrl,"titi",Pt2di(0,0),
                         MMIcone("AllW").in(1) *255
                         )
                     ),
    mCaseRollW      (new CaseGPUMT
                     (
                         *mPopUpCtrl,"titi",Pt2di(0,1),
                         MMIcone("RollW").in(1) *255
                         )
                     ),
    mCaseThisW       (new CaseGPUMT
                      (
                          *mPopUpCtrl,"titi",Pt2di(0,2),
                          MMIcone("ThisW").in(1) *255
                          )
                      ),
    mCaseThisPt      (new CaseGPUMT
                      (
                          *mPopUpCtrl,"titi",Pt2di(0,3),
                          MMIcone("ThisPt").in(1) *255
                          )
                      ),
    mCaseNewPt      (new CaseGPUMT
                     (
                         *mPopUp1Shift,"titi",Pt2di(0,1),
                         MMIcone("Pts").in(1) *255
                         )
                     ),
    mCaseKillPt      (new CaseGPUMT
                      (
                          *mPopUp1Shift,"titi",Pt2di(2,1),
                          MMIcone("TDM").in(1) *255
                          )
                      ),
    mCaseRenamePt    (new CaseGPUMT
                      (
                          *mPopUp1Shift,"titi",Pt2di(3,1),
                          MMIcone("Rename").in(1) *255
                          )
                      ),




    mCaseMin3       ( mUseMMPt ? new CaseGPUMT
                      (
                          *mPopUp1Shift,"titi",Pt2di(0,0),
                          MMIcone("Min3").in(1) *255
                          )
                        : 0) ,
    mCaseMin5        ( mUseMMPt ?  new CaseGPUMT
                      (
                          *mPopUp1Shift,"titi",Pt2di(1,0),
                          MMIcone("Min5").in(1) *255
                          )
                       : 0 ),
    mCaseMax3        (mUseMMPt ? new CaseGPUMT
                      (
                          *mPopUp1Shift,"titi",Pt2di(0,2),
                          MMIcone("Max3").in(1) *255
                          )
                       : 0),
    mCaseMax5        ( mUseMMPt ? new CaseGPUMT
                      (
                          *mPopUp1Shift,"titi",Pt2di(1,2),
                          MMIcone("Max5").in(1) *255
                          ) : 0
                      )
{
    if (anAppli.Param().Gama().Val() != 1.0)
        mVWV.SetGamaCorr(anAppli.Param().Gama().Val());
    aIm0.SetLoaded();
    SetImage(&aIm0);
}

void cWinIm::SetFullImage()
{
     mScr->set_max();
     ShowVect();
}

void  cWinIm::SetNoImage()
{
    if (mCurIm)
    {
#if (ELISE_X11)
        mScr->set_max();  // Modif MPD TENTATIVE CORRECTION BUG REAFF
#endif
        mCurIm->SetWAff(0);
    }
    mCurIm = 0;
}

void  cWinIm::SetImage(cImage *aIm)
{
    if (mCurIm)
    {
        mCurIm->SetWAff(0);
    }
    mCurIm = aIm;
    mCurIm->SetWAff(this);
    //  if (MPD_MM()) {std::cout << "AAAAAAAAAAAA\n";getchar();}
    mScr->set_max();
    //  if (MPD_MM()) {std::cout << "BBbbbbBBbbbb\n";getchar();}
    ShowVect();
    SetTitle();
}

void  cWinIm::SetNewImage(cImage * aIm)
{
    static int aCpt=0;
    aCpt++;
    aIm->CptAff() = aCpt;

    // std::cout << "OLD " << mCurIm->Name() << " NEW " << aIm->Name() << "\n";
#if (ELISE_X11)
    mScr->ReInitTifFile(aIm->Tif());
#endif
    SetImage(aIm);
}

Video_Win cWinIm::W()
{
    return mW;
}

bool  cWinIm::WVisible(const Pt2dr & aP)
{
    return (aP.x>0) && (aP.y>0) && (aP.x<mSzW.x) && (aP.y<mSzW.y);
}

bool cWinIm::PInIm(const Pt2dr & aP)
{
    Pt2di aSzIm = mCurIm->SzIm();
    return (aP.x>0) && (aP.y>0) && (aP.x<aSzIm.x) && (aP.y<aSzIm.y);
}

bool  cWinIm::WVisible(const Pt2dr & aP, eEtatPointeImage aState)
{
    return WVisible(aP) && mAppli.Interface()->Visible(aState);
}

bool  cWinIm::WVisible(cSP_PointeImage & aPIm)
{
    const cOneSaisie  & aSom = *(aPIm.Saisie());
    eEtatPointeImage aState = aSom.Etat();
    Pt2dr aP = aSom.PtIm();
    aP = mScr->to_win(aP);
    return    aPIm.Visible() && WVisible(aP, aState);
}

Box2dr  cWinIm::BoxImageVisible() const
{
    Box2dr aBoxIm(Pt2dr(0,0),Pt2dr(mCurIm->SzIm()));
    Box2dr aBoxWin(mScr->to_user(Pt2dr(0,0)),mScr->to_user(Pt2dr(mW.sz())));

    return Inf(aBoxIm,aBoxWin);
}


void  cWinIm::ShowVect()
{
    const std::vector<cSP_PointeImage *> &  aVP = mCurIm->VP();
    for (int aK=0 ; aK<int(aVP.size()) ; aK++)
    {
        if (WVisible(*(aVP[aK])))
        {
            const cOneSaisie  & aSom = *(aVP[aK]->Saisie());
            Pt2dr aP = aSom.PtIm();
            aP = mScr->to_win(aP);
            eEtatPointeImage aState = aSom.Etat();
            ShowPoint(aP,aState,aVP[aK]->Gl(),aVP[aK]);
        }
    }
}

void cWinIm::ShowPoint(const Pt2dr aP,eEtatPointeImage aState,cSP_PointGlob * aPG,cSP_PointeImage * aPIm)
{
    Elise_colour  aC = Elise_colour::yellow;
    if (aState == eEPI_Refute) aC = Elise_colour::red;
    if (aState == eEPI_Douteux) aC = Elise_colour::orange;
    if (aState == eEPI_Valide) aC = Elise_colour::green;

    Col_Pal aCP = mW.prgb()( (INT)(255*aC.r()), (INT)(255*aC.g()), (INT)(255*aC.b()) );
    Line_St aLst(aCP,2.0);

    mW.draw_circle_abs(aP,4,aLst);
    if (aPIm && mAppli.ShowDet())
    {
        ShowInfoPt(aPIm,false);
    }

    if (aPG && aPG->HighLighted())
    {

        std::vector<Pt2dr> aVPt;
        std::vector<bool>  aVOkPt;
        if (aPIm->BuildEpipolarLine(aVPt,aVOkPt))
        {
           std::vector<Pt2dr> aVPtW;
           Box2dr  aBoxIm = BoxImageVisible() ;
           for (int aK=1 ; aK<int(aVPt.size()) ; aK++)
           {
               if (aVOkPt[aK-1]||aVOkPt[aK])
               {
                   Seg2d aSeg(aVPt[aK-1],aVPt[aK]);
                   aSeg = aSeg.clip(aBoxIm);
                   if (! aSeg.empty())
                   {
                       Pt2dr aPW1 = mScr->to_win(aSeg.p0());
                       Pt2dr aPW2 = mScr->to_win(aSeg.p1());
                       mW.draw_seg(aPW1,aPW2,aLst);
                   }
               }
           }
        }
        else
        {
            mW.draw_circle_abs(aP,9,aLst);
        }
    }
}

void cWinIm::SetZoom(Pt2dr aP,double aFact)
{
    mScr->SetScArroundPW(aP,mScr->sc()*aFact,true);
}

void  cWinIm::GUR_query_pointer(Clik aClk,bool)
{
    // std::cout << "GST  : " << mCurIm->Tif().name() << " " << aClk._pt << "\n";
    if (mMode==eModeScale)
    {
        double aFact = pow(2.0,(aClk._pt.y-mLastPGrab.y)*0.01);
        SetZoom(mP0Grab,aFact);
        /*
        mScr->SetScArroundPW
        (
             mP0Grab,
             mScr->sc() *aFact,
             true
         );
*/
        mLastPGrab = aClk._pt;
    }
    else if (mMode==eModeTr)
    {
        mScr->SetDTrW(Pt2di((mLastPGrab-aClk._pt) ));
        mLastPGrab = aClk._pt;
    }
    else if (mMode==eModePopUp)
    {
        mPopUpCur->SetPtActif(Pt2di(aClk._pt));
    }
    else if (mMode==eModeSaisiePts)
    {
        mOldPt = mNewPt;
        mNewPt = aClk._pt;
        RedrawGrabSetPosPt();
    }
}

void cWinIm::GrabScrTr(Clik aClk)
{
    mMode= (aClk.shifted() ) ? eModeScale : eModeTr;
    mLastPGrab = aClk._pt;
    mP0Grab = mLastPGrab;
    mW.grab(*this);

    Redraw();
}

void cWinIm::SetTitle()
{
    ELISE_COPY(mWT.all_pts(),P8COL::yellow,mWT.odisc());
    mWT.fixed_string(Pt2dr(10,12),mCurIm->Name().c_str(),mWT.pdisc()(P8COL::black),true);
}

void cWinIm::Redraw()
{
    mScr->LoadAndVisuIm(mModeRelication);
    ShowVect();
    // mW.fixed_string(Pt2dr(10,20),mCurIm->Name().c_str(),mW.pdisc()(P8COL::black),true);
}


void cWinIm::RedrawGrabSetPosPt()
{
        // mOldPt = mNewPt;
        // mNewPt = aClk._pt;

    Pt2dr aPIm = mScr->to_user(mNewPt) ;


    // if (mCurIm->PtInImage(mNewPt))
    if ( !mCurIm->PtInImage(aPIm) )
    {
         mNewPt = mOldPt;
         return;
    }

    Pt2dr aPBox(5,5);
    ShowPoint(Pt2dr(mNewPt),mStatePtCur,0,0);
    mScr->LoadAndVisuIm(Pt2di(mOldPt-aPBox),Pt2di(mOldPt+aPBox),mModeRelication);
    ShowPoint(Pt2dr(mNewPt),mStatePtCur,0,0);
}


void cWinIm::ShowInfoPt(cSP_PointeImage * aPIm,bool Compl)
{
    Pt2dr aPt = mScr->to_win(aPIm->Saisie()->PtIm());
    mW.fixed_string(aPt+ Pt2dr(10,0),aPIm->Saisie()->NamePt().c_str(),mW.pdisc()(P8COL::black),true);
    std::string aNameType = eToString(aPIm->Gl()->PG()->Type());
    if (Compl)
    {
        aNameType = aNameType.substr(5,std::string::npos);
        mW.fixed_string(aPt+ Pt2dr(10,15),aNameType.c_str(),mW.pdisc()(P8COL::blue),true);
    }
}

void  cWinIm::SetPt(Clik aClk)
{
    if (aClk.shifted() && (!aClk.controled()))
    {
        MenuPopUp(aClk);
        return;
    }

    cSP_PointeImage * aPIm = GetNearest(aClk._pt,200);
    if (aPIm==0)
        return;


    if (aClk.shifted() && aClk.controled())
    {
        ShowInfoPt(aPIm,true);
        mW.disp().clik_release();
        Redraw();
        return;
    }

    if ((aPIm->Saisie()->Etat()== eEPI_Valide) &&  (!aClk.controled()))
    {
         return;
    }

    /*
   if ((!aClk.shifted()) && aClk.controled())
   {
        return;
   }
*/
    cPointGlob * aPG = aPIm->Gl()->PG();

    mMode = eModeSaisiePts;
    mOldPt = mScr->to_win(aPIm->Saisie()->PtIm());
    mStatePtCur = aPIm->Saisie()->Etat();
    mNewPt = aClk._pt;
    RedrawGrabSetPosPt();
    mW.grab(*this);

    if (!mCurIm->PtInImage(mScr->to_user(Pt2dr(mNewPt))))
    {
        Redraw();
        return;
    }

    if (! aClk.controled())
    {
        Pt2dr aP = FindPoint(mScr->to_user(mNewPt),aPG->Type(),aPG->SzRech().ValWithDef(3.0),aPG);
        if (aP==mAppli.Interface()->PtEchec)
            return;
        aP = mScr->to_win(aP);
        if (euclid(aP,mNewPt)>1e-3)
            ((cX11_Interface*) mAppli.Interface())->DrawZoom(mScr->to_user(aP));
        mNewPt = aP;
    }

    Pt2dr pt = mScr->to_user(Pt2dr(mNewPt));

    mAppli.Interface()->UpdatePoints(aPIm, pt);

    /*
   if (aPIm)
   {
         aPIm->Saisie()->PtIm() = mScr->to_user(aClk._pt);
         Reaff();
   }
*/
}

void  cWinIm::AffNextPtAct(Clik aClk)
{
    cSP_PointeImage * aPI = GetNearest(aClk._pt,1e9,true);

    if (! aPI) return;

    Pt2dr aP = aPI->Saisie()->PtIm();

    double aZoom = 20;
    mScr->set(aP -mSzW/(aZoom*2.0),aZoom);
    ShowVect();
}


cSP_PointeImage *  cWinIm::GetNearest(const Pt2dr & aPW,double aDSeuil,bool OnlyActif)
{
    aDSeuil = aDSeuil /mScr->sc();
    Pt2dr aPU =  mScr->to_user(aPW);
    const std::vector<cSP_PointeImage *> &  aVP = mCurIm->VP();

    cSP_PointeImage * aRes = 0;
    double aDMin = OnlyActif ? aDSeuil : 1e8;
    for (int aK=0; aK<int(aVP.size()) ; aK++)
    {
        cOneSaisie * aS = aVP[aK]->Saisie();
        const Pt2dr &  aP = aS->PtIm();
        // eEtatPointeImage aState = aS->Etat();
        bool Ok = WVisible(* aVP[aK]) ;
        double aDist = euclid(aPU,aP);
        if (OnlyActif)
        {
            Ok = true;
            switch (aS->Etat())
            {
                  case eEPI_NonSaisi :
                  break;

                  case eEPI_Douteux :
                     aDist += 1e5;
                  break;

                  case eEPI_Refute :
                     aDist += 2e5;
                  break;

                  default :
                     aDist = 1e10;
                  break;
            }
        }
        if ( Ok  && ( aDist < aDMin))
        {
            aDMin  = aDist;
            aRes = aVP[aK];
        }
    }
    return aRes;
}

BoolCaseGPUMT * cWinIm::BCaseVR()
{
    return mBCaseVisiRefut;
}

void  cWinIm::MenuPopUp(Clik aClk)
{

    mPopUpCur = 0;
    if (aClk._b==3)
    {
        if ((!aClk.controled()) && (!aClk.shifted())) mPopUpCur = mPopUpBase;
        if (aClk.shifted()   && (!aClk.controled()) ) mPopUpCur = mPopUpShift;
        if (aClk.controled() && (!aClk.shifted())   ) mPopUpCur = mPopUpCtrl;
    }

    if (aClk._b==1)
    {
        if (aClk.shifted()   && (!aClk.controled()) ) mPopUpCur = mPopUp1Shift;
    }

    if (mPopUpCur==0) return;

    mMode=eModePopUp;
    mPopUpCur->UpCenter(Pt2di(aClk._pt));
    mW.grab(*this);
    CaseGPUMT * aCase = mPopUpCur->PopAndGet();

    if (! aCase) return;


    if (mPopUpCur==mPopUp1Shift)
    {
        Redraw();
        if (aCase== mCaseNewPt)
        {
            CreatePoint(aClk._pt,eNSM_Pts,-1);
        }
        else if (aCase== mCaseMin3)
        {
            CreatePoint(aClk._pt,eNSM_MinLoc,3);
        }
        else if (aCase== mCaseMin5)
        {
            CreatePoint(aClk._pt,eNSM_MinLoc,5);
        }
        else if (aCase== mCaseMax3)
        {
            CreatePoint(aClk._pt,eNSM_MaxLoc,3);
        }
        else if (aCase== mCaseMax5)
        {
            CreatePoint(aClk._pt,eNSM_MaxLoc,5);
        }
        else if (aCase== mCaseKillPt)
        {
            cSP_PointeImage* aPIm = GetNearest(aClk._pt,200);
            if (aPIm)
            {
                ((cX11_Interface*)mAppli.Interface())->_DeletePoint(aPIm->Gl());
            }
        }
        else if (aCase== mCaseRenamePt)
        {
            cSP_PointeImage* aPIm = GetNearest(aClk._pt,200);
            if (aPIm)
            {
                cCaseNamePoint * aCNP = mAppli.Interface()->GetIndexNamePoint();
                if (aCNP && (aCNP->mTCP != eCaseCancel) && (aCNP->mFree))
                {
                    std::string aNewName = mAppli.Interface()->IdNewPts(aCNP).second;
                    mAppli.ChangeName(aPIm->Gl()->PG()->Name(),aNewName);
                }

                ((cX11_Interface*)mAppli.Interface())->MenuNamePoint()->W().lower();
            }
        }
    }

    if (mPopUpCur==mPopUpShift)
    {
        if (aCase== mCaseExit)
        {
            mAppli.Exit();
        }
        if (aCase==mCaseUndo)
        {
            mAppli.Undo();
        }
        if (aCase==mCaseRedo)
        {
            mAppli.Redo();
        }
        if (aCase== mBCaseVisiRefut)
        {
            mAppli.Interface()->SetInvisRef(mBCaseVisiRefut->Val());
        }
        if (aCase== mBCaseShowDet)
        {
            mAppli.ShowDet() = mBCaseShowDet->Val();
        }
        Redraw();
    }

    if (mPopUpCur==mPopUpBase)
    {
        eEtatPointeImage aState = eEPI_NonValue;
        if (aCase== mCaseVide)
            aState = eEPI_NonSaisi;
        if (aCase== mCaseTDM)
            aState = eEPI_Refute;
        if (aCase== mCaseInterrog)
            aState = eEPI_Douteux;
        if (aCase== mCaseSmile)
            aState = eEPI_Valide;
        if (aCase == mCaseHighLight)
        {
            cSP_PointeImage * aPIm = GetNearest(aClk._pt,200);
            if (aPIm)
            {
                mAppli.HighLightSom(aPIm-> Gl());
            }

            mAppli.RedrawAllWindows();
            return;
        }

        if (aState!=eEPI_NonValue)
        {
            cSP_PointeImage * aPIm = GetNearest(aClk._pt,200);
            if (aPIm)
            {
                mAppli.Interface()->ChangeState( aPIm, aState);
            }
        }
        Redraw();
    }

    if (mPopUpCur==mPopUpCtrl)
    {
        if (aCase==mCaseRollW)
        {
            mAppli.ChangeImages(0,((cX11_Interface*)mAppli.Interface())->WinIms(),true);
        }
        if (aCase==mCaseAllW)
        {
            mAppli.ChangeImages(0,((cX11_Interface*)mAppli.Interface())->WinIms(),false);
        }
        if (aCase==mCaseThisW)
        {
            std::vector<cWinIm *> aVWI;
            aVWI.push_back(this);
            mAppli.ChangeImages(0,aVWI,true);
        }
        if (aCase==mCaseThisPt)
        {
            cSP_PointeImage * aPIm = GetNearest(aClk._pt,200);
            if (aPIm)
            {
                mAppli.ChangeImages(aPIm->Gl(),((cX11_Interface*)mAppli.Interface())->WinIms(),false);
            }
        }
    }
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
