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


using namespace NS_SaisiePts;

cWinIm * cAppli_SaisiePts::WImOfW(Video_Win aW)
{
    for (int aK=0 ; aK<mNbW; aK++)
        if (mWins[aK]->W() == aW)
           return mWins[aK];

   return 0;
}

void cAppli_SaisiePts::TestClikWIm(Clik aCl)
{
  cWinIm * aWIm = WImOfW(aCl._w);
  if (!aWIm) 
     return;

  if (aCl._b==1)
  {
      aWIm->SetPt(aCl);
      Sauv();
  }

  if ((aCl._b==4) || (aCl._b==5))
  {
      double aFactZ = 1.2;
      aWIm->SetZoom(aCl._pt,(aCl._b==5) ? aFactZ: (1/aFactZ));
      aWIm->ShowVect();
  }




  if (aCl._b==2)
  {
      aWIm->GrabScrTr(aCl);
  }

  if (aCl._b==3)
  {
      aWIm->MenuPopUp(aCl);
  }
}

void cAppli_SaisiePts::BoucleInput()
{
   while(1)
   {
       Clik   aCl = mDisp->clik_press();

       TestClikWIm(aCl);
   }
}


void  cAppli_SaisiePts::SetInvisRef(bool aVal)
{
   mRefInvis = aVal;
   for (int aKW=0 ; aKW<int(mWins.size()) ; aKW++)
   {
         mWins[aKW]->BCaseVR()->SetVal(aVal);
         mWins[aKW]->Reaff();
         mWins[aKW]->ShowVect();
   }
}


void cAppli_SaisiePts::ReaffAllW()
{
    for (int aK=0 ; aK<int(mWins.size()) ; aK++)
        mWins[aK]->Reaff();
}

void cAppli_SaisiePts::UndoRedo(std::vector<cUndoRedo>  & ToExe ,std::vector<cUndoRedo>  & ToPush)
{
   if ( ToExe.empty())
      return;

   const cUndoRedo & anUR = ToExe.back();

   const cOneSaisie & aS = anUR.S();
   cSP_PointeImage * aPIm  = anUR.I()->PointeOfNameGlobSVP(aS.NamePt());
   ELISE_ASSERT(aPIm!=0,"Incoh in ExeUndoRedo");

   ToPush.push_back(cUndoRedo(*(aPIm->Saisie()),aPIm->Image()));
   *(aPIm->Saisie()) = aS;
   ToExe.pop_back();
   ReaffAllW();

}

void cAppli_SaisiePts::Undo()
{
    UndoRedo(mStackUndo, mStackRedo);
}
void cAppli_SaisiePts::Redo()
{
    UndoRedo(mStackRedo,mStackUndo);
}



void cAppli_SaisiePts::AddUndo(cOneSaisie aS,cImage * aI)
{

   mStackUndo.push_back(cUndoRedo(aS,aI));
   mStackRedo.clear();
}


const std::vector<cWinIm *> &  cAppli_SaisiePts::WinIms()
{
   return mWins;
}


bool cAppli_SaisiePts::Visible(cSP_PointeImage & aPIm)
{
    return   (aPIm.Saisie()->Etat() != eEPI_Refute)
           || mRefInvis;
}


void cAppli_SaisiePts::HighLightSom(cSP_PointGlob * aPG)
{
   for (int aKP=0 ; aKP<int(mPG.size()) ; aKP++)
   {
        if (mPG[aKP] == aPG)
          aPG->HighLighted() = ! aPG->HighLighted();
        else
          mPG[aKP]->HighLighted() = false;
   }
}

void cAppli_SaisiePts::ChangeName(std::string anOldName,std::string  aNewName)
{
    for (int aKP=0 ; aKP<int(mPG.size()) ; aKP++)
    {
         if (mPG[aKP]->PG()->Name() == aNewName)
         {
              std::cout << "Name " << aNewName << " already exist \n";
              return;
         }
    }

    for (int aKP=0 ; aKP<int(mPG.size()) ; aKP++)
    {
         if (mPG[aKP]->PG()->Name() == anOldName)
         {
             mPG[aKP]->Rename(aNewName);
         }
    }


    for (int aKC=0 ; aKC<int(mVNameCase.size()); aKC++)
    {
         cCaseNamePoint & aCN = mVNameCase[aKC];
         if (aCN.mTCP==eCaseStd)
         {
              if (aCN.mName == anOldName) 
              {
                  aCN.mFree = true;
              }
              if (aCN.mName == aNewName)
              {
                   aCN.mFree = false;
              }
         }
    }

    ReaffAllW();
}


   //================== cUndoRedo ==========

cUndoRedo::cUndoRedo(cOneSaisie aS,cImage *aI) :
   mS (aS),
   mI (aI)
{
}



const    cOneSaisie & cUndoRedo::S() const {return mS;}
cImage *              cUndoRedo::I() const {return mI;}




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
