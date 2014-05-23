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

cElHJaPlan3D::cElHJaPlan3D
(
        cElHJaArrangt &            anArrgt,
        INT                        aNum,
        const cElPlan3D &          aPlan,
        const tEmprPlani &         anEmprGlob,
        const std::vector<Pt2dr>&  anEmpriseSpec,
        Video_Win *                aW
) :
    mArrgt       (anArrgt),
    mNum         (aNum),
    mPlan        (aPlan),
    mEmpriseSpec (anEmpriseSpec),
    mHasEmprSpec (mEmpriseSpec.size() != 0),
    mW           (aW),
    mGr          (new tGrPl),
    mHasSegOblig (false)
{
    for (INT aK=0 ; aK<INT(anEmprGlob.size()) ; aK++)
       SomGrEmpr(anEmprGlob[aK]);
}

void cElHJaPlan3D::SetStatePlanInterdit()
{
     for (INT aK=0 ; aK<INT(mFacettes.size()) ; aK++)
         mFacettes[aK]->SetFacetteImpossible(0);
}

void cElHJaPlan3D::SetStatePlanWithSegOblig(tBufFacette * aBuf)
{
     for (INT aK=0 ; aK<INT(mFacOblig.size()) ; aK++)
     {
         mFacOblig[aK]->SetFacetteSure(aBuf);
         mFacOblig[aK]->PropageIncompVert(aBuf);
     }
}


void cElHJaPlan3D:: SetSegOblig(Seg2d aSeg)
{
    mHasSegOblig = true;
    mSegOblig = aSeg;
}

cElHJaPlan3D::~cElHJaPlan3D()
{
   delete mGr;
}

cElHJaArrangt & cElHJaPlan3D::Arrgt()
{
   return mArrgt;
}


tSomGrPl * cElHJaPlan3D::SomNearest(Pt2dr aP,REAL & aDMin)
{
   static tFullSubGrPl aSGrFul;
   aDMin = 1e9;
   tSomGrPl * aRes = 0;
   for (tItSomGrPl itS=mGr->begin(aSGrFul) ; itS.go_on() ; itS++)
   {
       REAL aDist = euclid(aP,(*itS).attr().Pt());
       if ((aDist < aDMin) || (aRes==0))
       {
          aDMin = aDist;
          aRes  = &(*itS);
       }
   }
   return aRes;
}

tSomGrPl *  cElHJaPlan3D::AddSom(bool & isNew, Pt2dr aP,REAL Absc,bool IsEmpr)
{
   static tFullSubGrPl aSGrFul;
   isNew = false;

   REAL aDist;
   tSomGrPl * aNearest =  SomNearest(aP,aDist);
   if (aDist < ElHJAEpsilon)
       return aNearest;

   isNew = true;
   return &(mGr->new_som(cElHJaAttrSomPlani(aP,Absc)));
}


tSomGrPl *  cElHJaPlan3D::SomGr3Pl(cElHJaPoint * aSom)
{
   bool isNew;
   return  AddSom(isNew,Proj(aSom->Pt()),0.0,false);
}

tSomGrPl *  cElHJaPlan3D::SomGrEmpr(const cElHJaSomEmpr & aSom)
{
   bool isNew;
   tSomGrPl * aRes = AddSom(isNew,aSom.Pos(),aSom.ACurv(),true);
   if (isNew)
       mVSomEmpr.push_back(aRes);
   return  aRes;
}



INT   cElHJaPlan3D:: Num() const             {return mNum;}
const cElPlan3D & cElHJaPlan3D::Plan() const {return mPlan;}

void cElHJaPlan3D::SetNbPlansInter(INT aNbPl)
{
     mVInters.clear();
     for (INT aK=0; aK<aNbPl ; aK++)
     {
          mVInters.push_back(0);
     }
}

void cElHJaPlan3D::AddInter(cElHJaDroite & anInter,cElHJaPlan3D & AutrPl)
{
    mVInters.at(AutrPl.Num())= &anInter;
}

cElHJaDroite * cElHJaPlan3D::DroiteOfInter(const cElHJaPlan3D & anOtherPl) const
{
   return mVInters.at(anOtherPl.Num());
}


void cElHJaPlan3D::Show
     (
          Video_Win aW,
          INT       aCoul,
          bool ShowDroite,
          bool ShowInterEmpr
     )
{
    if (aCoul >=0)
       ELISE_COPY(aW.all_pts(),aCoul,aW.ogray());

    Box2dr aBoxW(Pt2dr(0,0),Pt2dr(aW.sz()));
    for (INT aK=0; aK<INT(mVInters.size()) ; aK++)
    {
        cElHJaDroite * aDr =mVInters[aK];
	if (aDr)
	{
            ElSeg3D aSeg = aDr->Droite();
	    Pt3dr aQ0 = aSeg.P0();
	    Pt3dr aQ1 = aSeg.P1();
	    Pt2dr aP0(aQ0.x,aQ0.y);
	    Pt2dr aP1(aQ1.x,aQ1.y);

	    Seg2d aS(aP0,aP1);
            Seg2d aSC = aS.clipDroite(aBoxW);
            if (ShowDroite && (! aSC.empty()))
            {
	       aW.draw_seg(aSC.p0(),aSC.p1(),aW.pdisc()(P8COL::magenta));
            }
	}
    }

    tFullSubGrPl aSGrFul;
    if (ShowInterEmpr)
    {
        for (tItSomGrPl itS=mGr->begin(aSGrFul) ; itS.go_on() ; itS++)
	{
            aW.draw_circle_loc
            (
                (*itS).attr().Pt(),
                4.0,
                aW.pdisc()(P8COL::blue)
            );
	    for (tItArcGrPl itA=(*itS).begin(aSGrFul) ; itA.go_on() ; itA++)
	    {
                 tSomGrPl &s1 = (*itA).s1();
                 tSomGrPl &s2 = (*itA).s2();
		 if (&s1 < &s2)
		 {
                     aW.draw_seg
                     (
                         s1.attr().Pt(),
                         s2.attr().Pt(),
                        aW.pdisc()(P8COL::black)
                     );
		 }
	    }
	}
    }

    // for (INT aK=0 ; aK<INT(mFacOblig.size()) ; aK++)
    //    mFacOblig[aK]->Show(PI/2.0,P8COL::cyan,false);
}


class cCmpcElHJaSomEmpr_OnACurv
{
     public :
         bool operator ()
              (
                   const tSomGrPlPtr &aS1,
                   const tSomGrPlPtr &aS2
              )
         {
             return aS1->attr().ACurvE() < aS2->attr().ACurvE();
         }
};

tArcGrPl *  cElHJaPlan3D::NewArcInterieur(tSomGrPl *aS1,tSomGrPl * aS2)
{
	/*
   tArcGrPl  * aS1S2 = mGr->arc_s1s2(*aS1,*aS2);
   if (aS1S2)
      return aS1S2;
      */
    // cout << "NewArcInterieur \n";
    return &mGr->add_arc(*aS1,*aS2,cElHJaAttrArcPlani());
}

void cElHJaPlan3D::AddArcEmpriseInGraphe()
{
    cCmpcElHJaSomEmpr_OnACurv aCmp;
    std::sort(mVSomEmpr.begin(),mVSomEmpr.end(),aCmp);
    INT aNbS = (int) mVSomEmpr.size();

    for (INT aK=0 ; aK<aNbS ; aK++)
    {
        tSomGrPlPtr aS1 = mVSomEmpr[aK];
        tSomGrPlPtr aS2 = mVSomEmpr[(aK+1)%aNbS];
        mGr->add_arc(*aS1,*aS2,cElHJaAttrArcPlani());
    }
}


bool cElHJaPlan3D::MakeFacettes(cElHJaArrangt & anArgt)
{

   static cFullSubGrWithP   aSGrFul;
   ElPartition<tArcGrPl *>  aVFac;
   bool Ok = all_face_trigo(*mGr,aSGrFul,aVFac);


   for (INT aK=0 ; aK<aVFac.nb() ; aK++)
   {
       cElHJaFacette * aF = new cElHJaFacette(aVFac[aK].ToVect(),this);
       if (aF->IsExterne())
          delete aF;
       else
       {
          anArgt.AddFacette(aF);
	  mFacettes.push_back(aF);
       }
   }

   if (mHasSegOblig)
   {
       REAL aD;
       tSomGrPl * aS1 = SomNearest(mSegOblig.p0(),aD);
       tSomGrPl * aS2 = SomNearest(mSegOblig.p1(),aD);

       ElPcc<cElHJaAttrSomPlani,cElHJaAttrArcPlani> aPcc;
       cFullSubGrWithP  aSubSeg(mSegOblig);

       aPcc.pcc(*aS1,*aS2,aSubSeg,eModePCC_Somme);
       ElFilo<tSomGrPl *> aChem;
       aPcc.chemin(aChem,*aS2);

       INT aFlag = mGr->alloc_flag_arc();
       for (INT aK=0 ; aK<aChem.nb()-1 ; aK++)
           mGr->arc_s1s2(*aChem[aK],*aChem[aK+1])->sym_flag_set_kth_true(aFlag);

       for (INT aK=0 ; aK<INT(mFacettes.size()) ; aK++)
       {
            cElHJaFacette * aF = mFacettes[aK];
	    bool Got = false;
	    const  std::vector<tArcGrPl *>  & aVArcs = aF->Arcs();
            for (INT aK=0 ; aK<INT(aVArcs.size()) ; aK++)
                if (aVArcs[aK]->flag_kth(aFlag))
                    Got = true;
	    if (Got)
               mFacOblig.push_back(aF);
       }

       for (INT aK=0 ; aK<aChem.nb()-1 ; aK++)
           mGr->arc_s1s2(*aChem[aK],*aChem[aK+1])->sym_flag_set_kth_false(aFlag);
       mGr->free_flag_arc(aFlag);
   }

   return Ok;
}

Video_Win * cElHJaPlan3D::W() 
{
   return mW;
}

std::vector<std::vector<Pt3dr> > cElHJaPlan3D::FacesSols(bool WithIndet)
{
   static tFullSubGrPl aSGrFul;
   for (tItSomGrPl itS=mGr->begin(aSGrFul) ; itS.go_on() ; itS++)
       for (tItArcGrPl itA=(*itS).begin(aSGrFul) ; itA.go_on() ; itA++)
           (*itA).attr().ResetCpt();

   for (INT aKF=0 ; aKF<INT(mFacettes.size()) ; aKF++)
   {
       cElHJaFacette * aFac = mFacettes[aKF];
       const  std::vector<tArcGrPl *> & aVArc = aFac->Arcs();
       if (WithIndet)
       {
          if (! aFac->IsImpossible())
             for (INT aKA =0 ; aKA<INT(aVArc.size()) ; aKA++)
             {
                  aVArc[aKA]->attr().Set1();
                  aVArc[aKA]->arc_rec().attr().Set1();
             }
       }
       else
       {
          if (aFac->IsSure())
          {
             for (INT aKA =0 ; aKA<INT(aVArc.size()) ; aKA++)
             {
                  aVArc[aKA]->attr().IncrCpt();
                  aVArc[aKA]->arc_rec().attr().IncrCpt();
             }
          }
       }
   }

   std::vector<std::vector<Pt3dr> > aRes;
   cSubGrSol aSub;
   ElPartition<tArcGrPl *>  aVFac;
   all_face_trigo(*mGr,aSub,aVFac);

   for (INT aKF=0 ; aKF<aVFac.nb() ; aKF++)
   {
         std::vector<Pt3dr> aV3;
         std::vector<Pt2dr> aV2;
	 ElSubFilo<tArcGrPl *> aF = aVFac[aKF];
	 for (INT aKA=0 ; aKA<aF.nb() ; aKA++)
         {
             aV2.push_back(aF[aKA]->s1().attr().Pt());
             aV3.push_back(mPlan.AddZ(aF[aKA]->s1().attr().Pt()));
         }
        if (surf_or_poly(aV2) < 0)
            aRes.push_back(aV3);
   }

   return aRes;
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
