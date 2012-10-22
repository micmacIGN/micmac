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

cElHJaFacette::cElHJaFacette
(
      const std::vector<tArcGrPl *> & aCont,
      cElHJaPlan3D * aPlan
)  :
   mPlan (aPlan),
   mNbThisIsEnDessous (0)
{
     mStates.push_back(FBool::MayBe);
     Pt2dr aPMin(1e9,1e9);
     Pt2dr aPMax(1e-9,1e-9);
     for (INT aK=0 ; aK<INT(aCont.size()) ; aK++)
     {
          mVArcs.push_back(aCont[aK]);
	  Pt2dr aPt = mVArcs[aK]->s1().attr().Pt();
	  aPMin.SetInf(aPt);
	  aPMax.SetSup(aPt);
	  mVPt.push_back(aPt);
     }
     mSurf = surf_or_poly(mVPt);
     if(! IsExterne())
     {
        for (INT aK=0 ; aK<INT(aCont.size()) ; aK++)
	    aCont[aK]->attr().SetFacette(this);
     }
     mBox = Box2dr(aPMin,aPMax);
}

const  std::vector<tArcGrPl *>  & cElHJaFacette::Arcs()
{
    return mVArcs;
}

cElHJaArrangt & cElHJaFacette::Arrgt()
{
	return mPlan->Arrgt();
}

bool cElHJaFacette::IsExterne()
{
     return mSurf>0;
}

bool cElHJaFacette::PointInFacette(Pt2dr aP) const
{
   return PointInPoly(mVPt,aP);
}

cElHJaPlan3D * cElHJaFacette::Plan()
{
   return mPlan;
}

void cElHJaFacette::Show(REAL aDirH,INT aCoul,bool WithBox)
{
   Video_Win aW = *(mPlan->W());
   aW.hach(mVPt,Pt2dr::FromPolar(1.0,aDirH),3.0,aW.pdisc()(aCoul));
   if (WithBox)
   {
      aW.draw_rect(mBox._p0,mBox._p1,aW.pdisc()(aCoul));
   }
}


void cElHJaFacette::ShowCont(INT aCoul,Video_Win aW)
{
    INT aNb = (int) mVPt.size();
    for (INT aK=0 ; aK<aNb ; aK++)
        aW.draw_seg(mVPt[aK],mVPt[(aK+1)%aNb],aW.pdisc()(aCoul));
}

void cElHJaFacette::ShowGlob()
{
     Show(0.0,P8COL::green,false);

     for (INT aK=0 ; aK<INT(mVArcs.size()) ; aK++)
     {
         if (mVFAdjcPl[aK])
         {
            mVFAdjcPl[aK]->Show(1.0,P8COL::blue,false);
            mVFAdjcComp[aK]->Show(2.0,P8COL::red,false);
            mVFAdjcIncomp[aK]->Show(0.0,P8COL::cyan,false);
         }
     }
     for (INT aK=0 ; aK<INT(mVFRecouvrt.size()) ; aK++)
     {
         mVFRecouvrt[aK]->Show(PI/2,P8COL::black,false);
     }
}

void cElHJaFacette::MakeAdjacences()
{
     for (INT aK=0 ; aK<INT(mVArcs.size()) ; aK++)
     {
	     tArcGrPl * aArc1 = &(mVArcs[aK]->arc_rec());
	     tArcGrPl * aArc2 = mVArcs[aK]->attr().ArcHom();
	     tArcGrPl * aArc3  = (aArc2==0) ? 0 : &(aArc2->arc_rec());

	     cElHJaFacette * aF1 = aArc1->attr().Fac();
	     cElHJaFacette * aF3 = (aArc3==0) ? 0 :aArc3->attr().Fac();
	     cElHJaFacette * aF2 = (aArc2==0) ? 0 :aArc2->attr().Fac();
	     ELISE_ASSERT((aF1==0)==(aF3==0),"cElHJaFacette::MakeAdjacences");
	     ELISE_ASSERT((aF1==0)==(aF2==0),"cElHJaFacette::MakeAdjacences");
	     mVFAdjcPl.push_back(aF1);
	     mVFAdjcComp.push_back(aF3);
	     mVFAdjcIncomp.push_back(aF2);
     }
}

void  cElHJaFacette::AddRecouvrt(bool ThisIsEnDessus,cElHJaFacette * aF2)
{
     mVFRecouvrt.push_back(aF2);
     mThisIsEnDessus.push_back(ThisIsEnDessus);
     if (! ThisIsEnDessus)
       mNbThisIsEnDessous ++;
}

void cElHJaFacette::MakeRecouvrt(cElHJaFacette * aF2)
{
    bool ThisIsEnDessus;
    if (IsRecouvrt(ThisIsEnDessus,aF2,false))
    {
       AddRecouvrt(ThisIsEnDessus,aF2);
       aF2->AddRecouvrt(!ThisIsEnDessus,this);
    }
}
bool cElHJaFacette::IsRecouvrt(bool & ThisIsEnDessus,
                               cElHJaFacette * aF2,bool ShowMes)
{
     bool isIVide = InterVide(mBox,aF2->mBox);

     if (ShowMes)
        cout << "Inter " << isIVide << "\n";
     else
     {
        if (isIVide)
           return false;
     }

     cElPolygone aP1;
     cElPolygone aP2;
     aP1.AddContour(mVPt,false);
     aP2.AddContour(aF2->mVPt,false);

     cElPolygone aP3 = aP1 * aP2;
     cElPolygone::tConstLC  &  aLC = aP3.Contours();

     if (aLC.empty())
     {
        if (ShowMes)
           cout << "No Intersection \n";
        return false;
     }

     REAL aSTot = 0;
     REAL aPerimTot = 0;
     REAL aZ1 = 0.0;
     REAL aZ2 = 0.0;
     cElPlan3D PL1 = Plan()->Plan();
     cElPlan3D PL2 =  aF2->Plan()->Plan();
     for (cElPolygone::tItConstLC itC=aLC.begin(); itC!=aLC.end() ; itC++)
     {
         cElPolygone::tContour aC = *itC;
         REAL aS = surf_or_poly(aC);
         REAL aP = perimetre_poly(aC);
         if (ShowMes)
            cout << "P= " << aP << " ;  S= " << aS << "\n";
         aSTot     += ElAbs(aS);
         aPerimTot += ElAbs(aP);
         for (INT aKP=0; aKP<INT(aC.size()) ; aKP++)
         {
             aZ1 += PL1.ZOfXY(aC[aKP]);
             aZ2 += PL2.ZOfXY(aC[aKP]);
         }
     }

     ThisIsEnDessus = (aZ1>aZ2);

     REAL aDiam = aSTot / ElMax(aPerimTot,1e-5);

     if (ShowMes)
        cout << "Diam = " << aDiam << "\n";

     return aDiam > ElHJAEpsilon ;
}

INT   cElHJaFacette::NbThisEnDessous() const
{
   return mNbThisIsEnDessous;
}

void  cElHJaFacette::AddDessouSansDessus(std::vector<cElHJaFacette *> & aTabF)
{
   for (INT aK=0 ; aK<INT(mVFRecouvrt.size()) ; aK++)
   {
       if ( mThisIsEnDessus[aK])
       {
            mVFRecouvrt[aK]->mNbThisIsEnDessous--;
            if (mVFRecouvrt[aK]->mNbThisIsEnDessous==0)
               aTabF.push_back(mVFRecouvrt[aK]);
       }
   }
}


void cElHJaFacette::MakeInertie
     (
         Im2D_INT2 aImMnt,
	 INT aValForb,
	 cGenSysSurResol * aSys
     )
{
     if (! aSys)
        mMatInert = RMat_Inertie();
     Pt2di aSzIm = aImMnt.sz();
     INT aX0 = ElMax(0,round_down(mBox._p0.x));
     INT aY0 = ElMax(0,round_down(mBox._p0.y));
     INT aX1 = ElMax(aSzIm.x,round_up(mBox._p1.x));
     INT aY1 = ElMax(aSzIm.y,round_up(mBox._p1.y));
     INT2 ** aDMnt = aImMnt.data();

     for (INT anX=aX0 ; anX<aX1 ; anX++)
         for (INT anY=aY0 ; anY<aY1 ; anY++)
         {
            Pt2dr aPR(anX,anY);
            if  (PointInFacette(aPR))
            {
               INT aVal = aDMnt[anY][anX];
	       REAL aZFac = mPlan->Plan().ZOfXY(aPR);
	       if (aVal != aValForb)
	       {
                  if (aSys)
                      aSys->GSSR_Add_EqFitDroite(aZFac,aVal);
		  else
                      mMatInert.add_pt_en_place(aVal,aZFac);
	       }
            }
         }
}

const RMat_Inertie & cElHJaFacette::MatInert() const
{
   return mMatInert;
}

/********************************************************/
/*                                                      */
/*     Manipulation de l'etat des facettes              */
/*                                                      */
/********************************************************/

void cElHJaFacette::ShowState()
{
   if (TopState() == FBool::True)
      Show(0.0,P8COL::black,false);
   if (TopState() == FBool::MayBe)
      Show(0.0,P8COL::blue,false);
   if (TopState() == FBool::False)
      Show(0.0,P8COL::red,false);
}


void cElHJaFacette::DupState()
{
    mStates.push_back(mStates.back());
}

void cElHJaFacette::PopState()
{
    mStates.pop_back();
}

const FBool & cElHJaFacette::TopState() const
{
    return mStates.back();
}

bool cElHJaFacette::IsSure()
{
    return (TopState()==FBool::True);
}
bool cElHJaFacette::IsIndeterminee()
{
    return (TopState()==FBool::MayBe);
}

bool cElHJaFacette::IsImpossible()
{
    return (TopState()==FBool::False);
}

FBool & cElHJaFacette::TopState() 
{
    return mStates.back();
}

void cElHJaFacette::SetTopState(const FBool & aState)
{
    TopState() = aState;
}

bool cElHJaFacette::SetFacetteGen(tBufFacette * aBuf,const FBool& aNewState,const FBool& aForbidSate)
{
    if (TopState() == aForbidSate)
    {
        return true;
    }
   if (TopState() != aNewState)
   {
      Arrgt().AddStatistique(-ElAbs(mSurf),TopState());
      Arrgt().AddStatistique( ElAbs(mSurf),aNewState);
      if (aBuf)
         aBuf->push_back(this);
      TopState() = aNewState;
   }

   return false;
}

bool cElHJaFacette::SetFacetteSure(tBufFacette * aBuf)
{
     return SetFacetteGen(aBuf,FBool::True,FBool::False);
}

bool cElHJaFacette::SetFacetteImpossible(tBufFacette * aBuf)
{
     return SetFacetteGen(aBuf,FBool::False,FBool::True);
}


bool  cElHJaFacette::PropageIncompVertGen(tBufFacette * aBuf,bool OnlyDessous)
{
    if (TopState() != FBool::True)
       return false;

     for (INT aK=0 ; aK<INT(mVFRecouvrt.size()) ; aK++)
     {
         if (      (!OnlyDessous)
              ||   (mThisIsEnDessus[aK])
            )
         {
             if (mVFRecouvrt[aK]->SetFacetteImpossible(aBuf))
                return true;
         }
     }
     return false;
}

bool  cElHJaFacette::PropageIncompVert(tBufFacette * aBuf)
{
   return PropageIncompVertGen(aBuf,false);
}

bool  cElHJaFacette::PropageIncompEnDessous(tBufFacette * aBuf)
{
   return PropageIncompVertGen(aBuf,true);
}


bool  cElHJaFacette::PropageIcompVois(tBufFacette * aBuf)
{
    if (TopState() != FBool::False)
       return false;
     for (INT aK=0 ; aK<INT(mVArcs.size()) ; aK++)
     {
         if (
                mVFAdjcIncomp[aK] 
             && (mVFAdjcIncomp[aK]->TopState()==FBool::False)
         )
         {
              if (mVFAdjcPl[aK]->SetFacetteImpossible(aBuf) )
                  return true;
	      if (mVFAdjcComp[aK]->SetFacetteImpossible(aBuf))
                  return true;
	 }
     }
     return false;
           
}

bool cElHJaFacette::PropageVoisRecurs(tBufFacette * aBuf)
{
     while (! aBuf->empty())
     {
          if (aBuf->front()->PropageIcompVois(aBuf))
             return true;
	  aBuf->pop_front();
     }
     return false;
}

void cElHJaFacette::SetSureIfPossible()
{
     if (TopState() != FBool::MayBe)
         return;

     for (INT aK=0 ; aK<INT(mVFRecouvrt.size()) ; aK++)
     {
         if (mVFRecouvrt[aK]->TopState() != FBool::False)
            return;
     }

     // TopState() = FBool::True;
     SetFacetteSure(0);
     return;
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
