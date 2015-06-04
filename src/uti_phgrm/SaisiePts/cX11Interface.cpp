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

cX11_Interface::cX11_Interface(cAppli_SaisiePts &appli) :
    mCurWinIm     (0),
    mWZ           (0),
    mWEnter       (0)
{
    mParam = &appli.Param();
    mAppli = &appli;

    mRefInvis = appli.Param().RefInvis().Val();

    //  Init();
}

cX11_Interface::~cX11_Interface()
{
    delete mDisp;

    delete mCurWinIm;
    delete mWZ;
    delete mZFON;
    delete mMenuNamePoint;
    delete mWEnter;
}

void cX11_Interface::Init()
{

//SetImagesPriority(0,);

    InitNbWindows();

    Pt2di aSzF =  mParam->SectionWindows().SzTotIm().Val().dcbyc(mNb2W);;

    int aCpt=0;
    Video_Win * aLastW   = 0;
    Video_Win * aLastWX0 = 0;
    Video_Win * aWTitle  = 0;
    Video_Win * aWY0XMax = 0;
    Pt2di aSzTitle(aSzF.x,15);
    for (int aKY =0 ; aKY<mNb2W.y ; aKY++)
    {
        for (int aKX =0 ; aKX<mNb2W.x ; aKX++)
        {
            if (aCpt<mNbW)
            {
                Video_Win * aRes = 0;
                if (aCpt==0)
                {
                    aWTitle = Video_Win::PtrWStd(aSzTitle,true);
                    aRes = new Video_Win(*aWTitle,Video_Win::eBasG,aSzF);
                    // aRes = Video_Win::PtrWStd(aSzF,true);
                    aLastWX0 = aRes;
                    mDisp = new Video_Display(aRes->disp());
                }
                else
                {
                    if (aKX==0)
                    {
                        aWTitle = new Video_Win(*aLastWX0,Video_Win::eBasG,aSzTitle);
                        aRes = new Video_Win(*aWTitle,Video_Win::eBasG,aSzF);
                        aLastWX0 = aRes;
                    }
                    else
                    {
                        aWTitle = new Video_Win(*aLastW,Video_Win::eDroiteH,aSzTitle);
                        aRes = new Video_Win(*aWTitle,Video_Win::eBasG,aSzF);
                    }

                }
                ELISE_COPY(aWTitle->all_pts(),P8COL::yellow,aWTitle->odisc());
                ELISE_COPY(aRes->all_pts(),P8COL::blue,aRes->odisc());
                aLastW = aWTitle;
                mWins.push_back(new cWinIm(*mAppli,*aRes,*aWTitle,*mAppli->imageVis(aCpt)));
                if ((aKY==0) && (aKX==(mNb2W.x -1)))
                {
                    aWY0XMax = aWTitle;
                }
            }
            aCpt++;
        }
    }

    Pt2di zoomWindowSize = mParam->SectionWindows().SzWZ().ValWithDef(round_ni(Pt2dr(mParam->SzTotIm().Val())*0.6));
    mWZ =  new Video_Win(*aWY0XMax,Video_Win::eDroiteH, zoomWindowSize);
    mZFON = new cFenOuiNon(*mWZ,Pt2di(200,20));

    mVNameCase.push_back( cCaseNamePoint("Cancel",eCaseCancel) );

    if (mParam->EnterName().Val())
    {
        mVNameCase.push_back( cCaseNamePoint("Enter New",eCaseSaisie) );
    }

    InitVNameCase();

    mMenuNamePoint = new cFenMenu(*mWZ,Pt2di(120,20),Pt2di(1,mVNameCase.size()));

    if (mParam->EnterName().Val())
    {
        mWEnter =  new Video_Win(mMenuNamePoint->W(),Video_Win::eDroiteH,Pt2di(150,20));
        mWEnter->move_translate(Pt2di(0,20));
        ELISE_COPY(mWEnter->all_pts(),P8COL::yellow,mWEnter->odisc());
        //mWEnter->move_to(Pt2di(100,20));
        mWEnter->lower();
    }

    ELISE_COPY(mWZ->all_pts(),P8COL::green,mWZ->odisc());
}

cCaseNamePoint *  cX11_Interface::GetIndexNamePoint()
{
    Video_Win aW = mMenuNamePoint->W();
    aW.raise();

    for (int aK=0 ; aK<int(mVNameCase.size()) ; aK++)
    {
        int aGr = (aK%2) ? 255 : 200 ;
        Pt2di aPCase(0,aK);
        mMenuNamePoint->ColorieCase(aPCase,aW.prgb()(aGr,aGr,aGr),1);
        cCaseNamePoint & aCNP = mVNameCase[aK];

        mMenuNamePoint->StringCase(aPCase,aCNP.mFree ? aCNP.mName : "***" ,true);
    }

    Clik aClk = aW.clik_in();
    //aW.lower();

    Pt2di aKse = mMenuNamePoint->Pt2Case(Pt2di(aClk._pt));
    cCaseNamePoint * aRes =  &(mVNameCase[aKse.y]);

    //cout << aRes->mName << " Type "  << aRes->mTCP << endl;

    if (! aRes->mFree) return 0;

    return aRes;
}

void cX11_Interface::_DeletePoint(cSP_PointGlob * aSG)
{
    if (! mZFON->Get("Kill " + aSG->PG()->Name() + "?")) return;

    DeletePoint(aSG);

    RedrawAllWindows();
}

std::pair<int,std::string> cX11_Interface::IdNewPts(cCaseNamePoint * aCNP)
{
   int aCptMax = mAppli->GetCptMax() + 1;

   std::string aName = aCNP->mName;
   if (aCNP->mTCP == eCaseAutoNum)
   {
       aName = nameFromAutoNum(aCNP, aCptMax);
   }

   if (aCNP->mTCP == eCaseSaisie)
   {
       mWEnter->raise();
       ELISE_COPY(mWEnter->all_pts(),P8COL::yellow,mWEnter->odisc());

       // std::cin >> aName ;
       aName = mWEnter->GetString(Pt2dr(5,15),mWEnter->pdisc()(P8COL::black),mWEnter->pdisc()(P8COL::yellow));
       mWEnter->lower();
   }

   mMenuNamePoint->W().lower();

   // std::cout << "cAppli_SaisiePts::IdNewPts " << aCptMax << " " << aName << "\n";

   return std::pair<int,std::string>(aCptMax,aName);
}

void cX11_Interface::DrawZoom(const Pt2dr & aPGlob)
{
     double aZoom = 10.0;

     Pt2dr aPIm = aPGlob- Pt2dr(mAppli->DecRech());
     Pt2dr aPMil = Pt2dr(mWZ->sz())/(2.0*aZoom);

     Video_Win aWC = mWZ->chc(aPIm-aPMil,Pt2dr(aZoom,aZoom));
     ELISE_COPY
     (
                aWC.all_pts(),
                mAppli->ImRechVisu().in(0),
                aWC.ogray()
     );

     aWC.draw_circle_abs(aPIm,4.0,Line_St(aWC.pdisc()(P8COL::blue),3.0));
}

cWinIm * cX11_Interface::WinImOfW(Video_Win aW)
{
    for (int aK=0 ; aK<mNbW; aK++)
        if (mWins[aK]->W() == aW)
           return mWins[aK];

    return 0;
}

void cX11_Interface::TestClick(Clik aCl)
{
    cWinIm * aWIm = WinImOfW(aCl._w);
    if (!aWIm)
        return;
    else
        mCurWinIm = aWIm;

    if (aCl._b==1)
    {
        aWIm->SetPt(aCl);
        Save();
    }

    if ((aCl._b==4) || (aCl._b==5))
    {
        double aFactZ = 1.2;
        aWIm->SetZoom(aCl._pt,(aCl._b==5) ? aFactZ: (1/aFactZ));
        aWIm->ShowVect();
    }

    if (aCl._b==2)
    {
        if (aCl.controled() )
        {
            // std::cout << "Zzzz2222 " << aCl.controled() << "\n";
           if (aCl.shifted())
           {
              mCurWinIm->SetFullImage();
           }
           else
           {
              mCurWinIm->AffNextPtAct(aCl);
           }
        }
        aWIm->GrabScrTr(aCl);
    }

    if (aCl._b==3)
    {
        aWIm->MenuPopUp(aCl);
    }
}

void cX11_Interface::BoucleInput()
{
    while(1)
    {
        Clik   aCl = mDisp->clik_press();

        TestClick(aCl);
    }
}

void cX11_Interface::SetInvisRef(bool aVal)
{
    mRefInvis = aVal;

    for (int aKW=0 ; aKW < int(mWins.size()); aKW++)
    {
        mWins[aKW]->BCaseVR()->SetVal(aVal);
        mWins[aKW]->Redraw();
        mWins[aKW]->ShowVect();
    }
}

void cX11_Interface::AddUndo(cOneSaisie * aSom)
{
    mAppli->AddUndo(*aSom, mCurWinIm->Image());
}

bool cX11_Interface::isDisplayed(cImage *anIm)
{
    return anIm->WAff() != NULL;
}

void cX11_Interface::Warning(std::string aMsg)
{
    std::cout << aMsg;
}

void cX11_Interface::RedrawAllWindows()
{
    for (int aK=0 ; aK< int(mWins.size()) ; aK++)
        mWins[aK]->Redraw();
}

void cX11_Interface::Redraw()
{
    mCurWinIm->Redraw();
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
