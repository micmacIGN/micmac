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

/***************************************************/
/***************************************************/
/***                                             ***/
/***    GridPopUpMenuTransp                      ***/
/***                                             ***/
/***************************************************/
/***************************************************/


GridPopUpMenuTransp::GridPopUpMenuTransp
(
     Video_Win aW,
     Pt2di aSzGrid,
     Pt2di aNbGrid,
     Pt2di aBrdGrid
)   :
    PopUpMenuTransp
    (
       aW,
       aSzGrid.mcbyc(aNbGrid)+aBrdGrid*2
    ),
    mSzGrid    (aSzGrid),
    mNbGrid    (aNbGrid),
    mBrdGrid   (aBrdGrid),
    mGridIsAct (false),
    mCases     (aNbGrid.y,std::vector<CaseGPUMT *>(aNbGrid.x,(CaseGPUMT *)NULL)),
    mCurCase   (0)
{
    for (INT iGrY= 0; iGrY<mNbGrid.y ; iGrY++)
        for (INT iGrX= 0; iGrX<mNbGrid.x ; iGrX++)
            SetBorderMotif(Pt2di(iGrX,iGrY));
}

void GridPopUpMenuTransp::SetBorderMotif(Pt2di iGrid)
{
    ELISE_COPY
    (
        border_rect
        (
            Indice2P0Grid(iGrid),
            Indice2P1Grid(iGrid)-Pt2di(1,1)
        ),
        0,
        mImMotif.oclip()
    );
}

Pt2di GridPopUpMenuTransp::Indice2P0Grid(Pt2di iGrid) const
{
   return mBrdGrid + mSzGrid.mcbyc(iGrid);
}

Pt2di GridPopUpMenuTransp::Indice2P1Grid(Pt2di iGrid) const
{
    return Indice2P0Grid(iGrid) +  mSzGrid;
}

Pt2di GridPopUpMenuTransp::Pt2Indice(Pt2di aPt) const
{
    return  Sup
            (
               Pt2di(0,0),
               Inf
               ( 
                   mNbGrid-Pt2di(1,1),
                   (aPt-mBrdGrid).dcbyc(mSzGrid)
               )
            );
}



CaseGPUMT *  GridPopUpMenuTransp::PopAndGet()
{
   PopUpMenuTransp::Pop();
   mGridIsAct = false;
   CaseGPUMT * res = mCurCase;
   mCurCase = 0;
   if (res) 
      res->CaseGPUMTOnSelected();
   return  res;
}

void GridPopUpMenuTransp::compute_menu_actif(Pt2di p0,Pt2di p1)
{
    for (INT y=p0.y ; y<p1.y; y++)
        for (INT x=p0.x ; x<p1.x; x++)
        {
            if (mDataM[y][x])
            {
               mDataR[y][x] = 255;
               mDataG[y][x] = 0;
               mDataB[y][x] = 0;
            }
            else
            {
               mDataR[y][x] = 0;
               mDataG[y][x] = 0;
               mDataB[y][x] = 255;
            }
        }

}

void GridPopUpMenuTransp::EffaceCaseGridActif(Pt2di iGrid)
{
     if (! mGridIsAct) 
          return;
     mGridIsAct = false;
     Pt2di  P0 = Indice2P0Grid(iGrid);
     Pt2di  P1 = Indice2P1Grid(iGrid);
     load_compute_show_rgb(P0+mP0Cur,P0,P1-P0);
     if (mCurCase) 
        mCurCase->CaseGPUMTOffActif();
    mCurCase = 0;
}

void GridPopUpMenuTransp::VisuCaseGridActif(Pt2di iGrid)
{
     Pt2di  P0 = Indice2P0Grid(iGrid);
     Pt2di  P1 = Indice2P1Grid(iGrid);
     compute_menu_actif(P0,P1);
     show_rgb(P0,P0+mP0Cur,P1-P0);
}




void GridPopUpMenuTransp::SetPtActif(Pt2di pt)
{
    if (!PtInsideMenu(pt))
    {
        EffaceCaseGridActif(mIndGrAct);
        return;
    }

    pt -= mP0Cur;
    {
       Pt2di iGr = Pt2Indice(pt);

       if (mGridIsAct)
       {

          if (mIndGrAct==iGr) 
              return;
           EffaceCaseGridActif(mIndGrAct);
       }

       mGridIsAct = true;
       mIndGrAct  = iGr;
   }

   mCurCase = mCases[mIndGrAct.y][mIndGrAct.x] ;
   if (mCurCase) 
      mCurCase->CaseGPUMTOnActif();

   VisuCaseGridActif(mIndGrAct);
}


/*
void GridPopUpMenuTransp::SetCase(CaseGPUMT * aCase,Pt2di anInd)
{
    ELISE_ASSERT
    (
        anInd.in_box(Pt2di(0,0),mNbGrid),
       "Out Of Inexe in GridPopUpMenuTransp::SetCase"
    );

    mCases[anInd.y][anInd.x] = aCase;
    aCase->mGPUMT = this;
    aCase->mInd = anInd;
}
*/


/***************************************************/
/***************************************************/
/***                                             ***/
/***    CaseGPUMT                                ***/
/***                                             ***/
/***************************************************/
/***************************************************/

void CaseGPUMT::SetFonc(Fonc_Num aFonc,Im2D_U_INT1 anIm)
{
    ELISE_COPY
    (
        rectangle
        (
            mGPUMT.Indice2P0Grid(mInd),
            mGPUMT.Indice2P1Grid(mInd)
        ),
        trans(aFonc,-mGPUMT.Indice2P0Grid(mInd)),
        anIm.out()
    );
}

void CaseGPUMT::SetMotif(Fonc_Num aFonc) 
{
    SetFonc(aFonc,mGPUMT.mImMotif);
}

void CaseGPUMT::SetTransp(Fonc_Num aFonc) 
{
    SetFonc(aFonc,mGPUMT.mImTransp);
}




CaseGPUMT::CaseGPUMT
(
      GridPopUpMenuTransp & aGPUMT,
      const std::string & aName ,
      Pt2di anInd,
      Fonc_Num anInitFonc
) :
      mGPUMT (aGPUMT),
      mName  (aName),
      mInd   (anInd)
{
    ELISE_ASSERT
    (
         aGPUMT.mCases[anInd.y][anInd.x]==0,
         "Multiple Affectation of CaseGPUMT"
    );
    aGPUMT.mCases[anInd.y][anInd.x] = this;
    SetMotif(anInitFonc);
}

void CaseGPUMT::CaseGPUMTOnActif()
{
}

void CaseGPUMT::CaseGPUMTOffActif()
{
}

void CaseGPUMT::CaseGPUMTOnSelected()
{
}

CaseGPUMT::~CaseGPUMT() {}


/***************************************************/
/***************************************************/
/***                                             ***/
/***    BoolCaseGPUMT                            ***/
/***                                             ***/
/***************************************************/
/***************************************************/


BoolCaseGPUMT::BoolCaseGPUMT
(
    GridPopUpMenuTransp & aGridMenu,
    const std::string & aName,
    Pt2di anInd,
    Fonc_Num aFTrue,
    Fonc_Num aFFalse,
    bool     aVal
)  :
   CaseGPUMT      (aGridMenu,aName,anInd,aVal ? aFTrue: aFFalse),
   mVal           (aVal),
   mImTrue        (aGridMenu.SzGrid().x,aGridMenu.SzGrid().y),
   mImFalse       (aGridMenu.SzGrid().x,aGridMenu.SzGrid().y)
{
   ELISE_COPY(mImTrue.all_pts() ,aFTrue, mImTrue.out() );
   ELISE_COPY(mImFalse.all_pts(),aFFalse,mImFalse.out());
}


void BoolCaseGPUMT::CaseGPUMTOnSelected()
{
   SetVal(! mVal);
}

void  BoolCaseGPUMT::SetVal(bool aVal)
{
    mVal = aVal;
    SetMotif(mVal ? mImTrue.in(0) : mImFalse.in(0));
    BoolCaseGPUMTOnChState(mVal);
}

void BoolCaseGPUMT::BoolCaseGPUMTOnChState(bool)
{
}

bool BoolCaseGPUMT::Val() const
{
   return mVal;
}

/***************************************************/
/***************************************************/
/***                                             ***/
/***    ChoixParmiCaseGPUMT                      ***/
/***                                             ***/
/***************************************************/
/***************************************************/

ChoixParmiCaseGPUMT::ChoixParmiCaseGPUMT
(
     GridPopUpMenuTransp & aGridMenu,
     const std::string & aName,
     Pt2di anInd,
     Fonc_Num aFTrue,
     Fonc_Num aFFalse,
     INT      anId,
     ChoixParmiCaseGPUMT * oneExcluing  // 0 pour la premiere
)   :
   CaseGPUMT      (aGridMenu,aName,anInd,(oneExcluing==0) ? aFTrue: aFFalse),
   mId            (anId),
   mSelected      (oneExcluing ? oneExcluing->mSelected : new (ChoixParmiCaseGPUMT *)),
   mImTrue        (aGridMenu.SzGrid().x,aGridMenu.SzGrid().y),
   mImFalse       (aGridMenu.SzGrid().x,aGridMenu.SzGrid().y)
{
   if (oneExcluing ==0) 
      *mSelected = this;
   ELISE_COPY(mImTrue.all_pts() ,aFTrue, mImTrue.out() );
   ELISE_COPY(mImFalse.all_pts(),aFFalse,mImFalse.out());
}

void ChoixParmiCaseGPUMT::CaseGPUMTOnSelected()
{
    if (*mSelected == this)
       return;

     (*mSelected)->ChoixParmiCaseGPUMTOnChState(false);
     (*mSelected)->SetMotif((*mSelected)->mImFalse.in(0));

     ChoixParmiCaseGPUMTOnChState(true);
     SetMotif(mImTrue.in(0));

     *mSelected = this;
}

void ChoixParmiCaseGPUMT::ChoixParmiCaseGPUMTOnChState(bool)
{
}

const ChoixParmiCaseGPUMT * ChoixParmiCaseGPUMT::Selected() const
{
   return * mSelected;
}

INT  ChoixParmiCaseGPUMT::IdSelected() const
{
   return (* mSelected)->mId;
}

/***************************************************/
/***************************************************/
/***                                             ***/
/***    PopUpMenuTransp                          ***/
/***                                             ***/
/***************************************************/
/***************************************************/


PopUpMenuTransp::PopUpMenuTransp
(
    Video_Win aW,
    Pt2di aSz
) :
   mSz       (aSz),
   mW        (aW),
   mWStdIm   (aW.StdBigImage()),
   mWAffIm   (mW,aSz),
   mImMotif  (mSz.x,mSz.y,1),
   mImTransp (mSz.x,mSz.y,128),
   mImR      (mSz.x,mSz.y),
   mImG      (mSz.x,mSz.y),
   mImB      (mSz.x,mSz.y),
   mDataM    (mImMotif.data()),
   mDataT    (mImTransp.data()),
   mDataR    (mImR.data()),
   mDataG    (mImG.data()),
   mDataB    (mImB.data()),
   mUp       (false)
{
    ELISE_COPY
    (
        mImMotif.border(1),
        0,
        mImMotif.out()
    );
}

PopUpMenuTransp::~PopUpMenuTransp()
{
}


void PopUpMenuTransp::AdaptP0(Pt2di & p0Im)
{
    p0Im = Sup(p0Im,Pt2di(0,0));
    p0Im = Inf(p0Im,mW.sz()-mSz);
}

void PopUpMenuTransp::UpP0(Pt2di p0Im)
{
    AdaptP0(p0Im);
    mUp = true;

    load_compute_show_rgb(p0Im,Pt2di(0,0),mSz);

    mP0Cur = p0Im;

}

void PopUpMenuTransp::load_compute_show_rgb(Pt2di P0Src,Pt2di P0Dest,Pt2di sz)
{
    load_rgb(P0Src,P0Dest,sz);
    compute_rgb(P0Dest,P0Dest+sz);
    show_rgb(P0Dest,P0Src,sz);
}

void PopUpMenuTransp::UpCenter(Pt2di p0Im)
{
    UpP0(p0Im-(mSz-Pt2di(30,20))/2);
}


void PopUpMenuTransp::load_rgb(Pt2di P0Src,Pt2di P0Dest,Pt2di sz)
{
    mWStdIm.read_in_el_image
    (
       P0Src,P0Dest,sz,
       mImR,mImG,mImB
    ); 
}


void PopUpMenuTransp::show_rgb(Pt2di P0Src,Pt2di P0Dest,Pt2di sz)
{
    mWAffIm.fill_with_el_image
    (
       P0Src,Pt2di(0,0),sz,
       mImR,mImG,mImB
    ); 

    mWAffIm.load(Pt2di(0,0),P0Dest,sz);

}

void PopUpMenuTransp::compute_rgb(Pt2di p0,Pt2di p1)
{
    p0 = Sup(p0,Pt2di(0,0));
    p1 = Inf(p1,mSz);


    for (INT y=p0.y ; y<p1.y; y++)
        for (INT x=p0.x ; x<p1.x; x++)
        {
            INT tR = 196;
            INT tV = 64;
            INT tB = 128;
            // INT  mOp = mDataM[y][x] * (256-t); // mOp = motif * opacite
 
            mDataR[y][x] = (mDataR[y][x]*tR+mDataM[y][x] * (256-tR))>>8;
            mDataG[y][x] = (mDataG[y][x]*tV+mDataM[y][x] * (256-tV))>>8;
            mDataB[y][x] = (mDataB[y][x]*tB+mDataM[y][x] * (256-tB))>>8;
        }

/*

    for (INT y=p0.y ; y<p1.y; y++)
        for (INT x=p0.x ; x<p1.x; x++)
        {
            INT t = 196;
            INT  mOp = mDataM[y][x] * (256-t); // mOp = motif * opacite
 
            mDataR[y][x] = (mDataR[y][x]*t+mOp)>>8;
            mDataG[y][x] = (mDataG[y][x]*t+mOp)>>8;
            mDataB[y][x] = (mDataB[y][x]*t+mOp)>>8;
        }
    for (INT y=p0.y ; y<p1.y; y++)
        for (INT x=p0.x ; x<p1.x; x++)
        {
            mDataG[y][x] =   (255-mDataM[y][x]) ;
       }


    for (INT y=p0.y ; y<p1.y; y++)
        for (INT x=p0.x ; x<p1.x; x++)
        {
            INT t = mDataT[y][x];
            INT  mOp = mDataM[y][x] * (256-t); // mOp = motif * opacite
 
            mDataR[y][x] = (mDataR[y][x]*t+mOp)>>8;
            mDataG[y][x] = (mDataG[y][x]*t+mOp)>>8;
            mDataB[y][x] = (mDataB[y][x]*t+mOp)>>8;
        }

    for (INT y=p0.y ; y<p1.y; y++)
        for (INT x=p0.x ; x<p1.x; x++)
        {
            INT Gr = (mDataR[y][x]+ mDataG[y][x]+mDataB[y][x])/3;
 
            mDataR[y][x] = Gr;
            mDataG[y][x] = mDataM[y][x];
            mDataB[y][x] = mDataM[y][x];
        }

    for (INT y=p0.y ; y<p1.y; y++)
        for (INT x=p0.x ; x<p1.x; x++)
        {
            bool BlackFond = ( mDataR[y][x]<128);
            bool BlackMotif = ( mDataM[y][x] != 0);

            if (BlackFond&&BlackMotif)
            {
                mDataR[y][x] =  0;
                mDataG[y][x] =  0;
                mDataB[y][x] =  0;
            }
            if (BlackFond&&(!BlackMotif))
            {
                mDataR[y][x] =  255;
                mDataG[y][x] =  196;
                mDataB[y][x] =  255;
            }
            if ((!BlackFond)&&BlackMotif)
            {
                mDataR[y][x] =  0;
                mDataG[y][x] =  128;
                mDataB[y][x] =  0;
            }
            if ((!BlackFond)&&(!BlackMotif))
            {
                mDataR[y][x] =  255;
                mDataG[y][x] =  255;
                mDataB[y][x] =  255;
            }
       }
*/
}

Box2di PopUpMenuTransp::BoxInFen() const
{
   return Box2di(mP0Cur,mSz+mP0Cur);
}

void PopUpMenuTransp::reset_im_ori(Pt2di p0,Pt2di p1)
{
    mW.load_image(p0,p0,p1-p0);
}

void PopUpMenuTransp::Pop()
{
   if (!mUp)
      return;

   mUp = false;
   reset_im_ori(mP0Cur,mSz+mP0Cur);
}

Pt2di PopUpMenuTransp::P0Cur() const
{
   return mP0Cur;
}

bool PopUpMenuTransp::PtInsideMenu(Pt2di pt)
{
    return pt.in_box(mP0Cur,mSz+mP0Cur);
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
