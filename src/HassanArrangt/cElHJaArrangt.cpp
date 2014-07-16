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


/*****************************************************/
/*                                                   */
/*                 cStatistique                      */
/*                                                   */
/*****************************************************/

cElHJaArrangt::cStatistique::cStatistique
(
     REAL aSurfSure,
     REAL aSurfImpos,
     REAL aSurfIndet
)  :
   mSurfSure        (aSurfSure),
   mSurfImpossible  (aSurfImpos),
   mSurfIndet       (aSurfIndet)
{
}

/*****************************************************/
/*                                                   */
/*                 cElHJaArrangt                     */
/*                                                   */
/*****************************************************/

cElHJaArrangt::cElHJaArrangt
(
    Pt2di aNbVisu
)  :
   mVisu (
            (aNbVisu.x>=0 ) ? 
	    new cElHJaArrangt_Visu(Pt2di(400,300),aNbVisu,Pt2di(200,150)) : 
	    0
	 ),
   mNbPl (-1),
   mSysResolv (0),
   mL2SysResolv (2),
   mL1SysResolv (2,100000)
{
}

Video_Win * cElHJaArrangt::WinOfPl(INT aK)
{
	return mVisu  ? mVisu->WinOfPl(aK) : 0;
}


void cElHJaArrangt::ReInit(const std::vector<Pt2dr> & anEmprise)
{
     mEmprVPt2d = anEmprise;
     mEmprPl.clear();
     for (INT aK=0 ;aK<INT(anEmprise.size()) ; aK++)
         mEmprPl.push_back
         (
            cElHJaSomEmpr
            (
                anEmprise[aK],
                (aK==0) ? 0 : &(mEmprPl.back())
            )
         );

     DeleteAndClear(mPlans);
     DeleteAndClear(mDroites);
     DeleteAndClear(mPoints);
     DeleteAndClear(mFacettes);
     mNbPl = 0;
     mStats.clear();
     mSurfEmpr = ElAbs(surf_or_poly(mEmprVPt2d));
}

void cElHJaArrangt::AddFacette(cElHJaFacette * aFac)
{
     mFacettes.push_back(aFac);
}


cElHJaPlan3D * cElHJaArrangt::AddPlan
     (
           const cElPlan3D & aPlan,
	   const std::vector<Pt2dr>* anEmprise
     )
{
     mPlans.push_back
     (
           new cElHJaPlan3D
	   (
               *this,
	       mNbPl,
	       aPlan,
               mEmprPl,
               (anEmprise ? *anEmprise : mEmptyEmprise),
	       WinOfPl(mNbPl)
	   )
     );
     mNbPl++;
     return mPlans.back();
}

const REAL cElHJaArrangt::Epsilon = 1e-4;


void cElHJaArrangt::ConstruireAll()
{
     mStats.push_back
     (
         cStatistique
	 (
	      0.0,
	      0.0,
	      mSurfEmpr * mNbPl
	 )
     );    
     bool toShoTime = false;
     ElTimer aChrono;
     ELISE_ASSERT(mNbPl>=0,"No Init int cElHJaArrangt");

     // Construction de la geometrie 3D
     for (tItPl itPl = mPlans.begin() ; itPl!=mPlans.end() ; itPl++)
     {
          (*itPl)->SetNbPlansInter(mNbPl);
     }

     for (tItPl itPl1 = mPlans.begin() ; itPl1!=mPlans.end() ; itPl1++)
     {
          for (tItPl itPl2 =  NextIter(itPl1) ; itPl2!=mPlans.end() ; itPl2++)
	  {
              bool Ok;
              ElSeg3D aD3d = (*itPl1)->Plan().Inter((*itPl2)->Plan(),Ok);
              if (Ok)
              {
                 mDroites.push_back(new cElHJaDroite(aD3d,**itPl1,**itPl2,mNbPl));
                 for (tItPl itPl3=NextIter(itPl2) ; itPl3!=mPlans.end() ; itPl3++)
	         {
                      Pt3dr aP =  (*itPl1)->Plan().Inter((*itPl2)->Plan(),(*itPl3)->Plan(),Ok);
                      if (Ok && PointInPoly(mEmprVPt2d,Proj(aP)))
                         mPoints.push_back(new cElHJaPoint(aP,**itPl1,**itPl2,**itPl3));
	         }
              }
	  }
     }

     for (tItPoint itPt = mPoints.begin() ; itPt!=mPoints.end() ; itPt++)
         (*itPt)->MakeDroites();

    if (toShoTime)
        cout << "Time Geom3D : " << aChrono.ValAndInit() << "\n";


     // Construction des Intersection Droites/emprise
     for (tItDr itD = mDroites.begin(); itD!=mDroites.end(); itD++)
         (*itD)->MakeIntersectionEmprise(mEmprPl);

    if (toShoTime)
        cout << "Time Inter  Emprise : " << aChrono.ValAndInit() << "\n";

     // Construction des facettes dans chaque plan
     for (tItPl itPl = mPlans.begin() ; itPl!=mPlans.end() ; itPl++)
         (*itPl)->AddArcEmpriseInGraphe();

     for (tItDr itD = mDroites.begin(); itD!=mDroites.end(); itD++)
         (*itD)->AddArcsInterieurInGraphe(mEmprVPt2d);

     for (tItPl itPl = mPlans.begin() ; itPl!=mPlans.end() ; itPl++)
         (*itPl)->MakeFacettes(*this);
     
    if (toShoTime)
        cout << "Time Facette  : " << aChrono.ValAndInit() << "\n";

     // Construction des  relations entre facettes
     
     for (tItFac itF = mFacettes.begin() ; itF!=mFacettes.end() ; itF++)
	 (*itF)->MakeAdjacences();

     for (tItFac itF1 = mFacettes.begin() ; itF1!=mFacettes.end() ; itF1++)
         for (tItFac itF2 =  NextIter(itF1) ; itF2!=mFacettes.end() ; itF2++)
             (*itF1)->MakeRecouvrt(*itF2);

    TriTopologiqueFacette();

    if (toShoTime)
        cout << "Time Relation F  : " << aChrono.ValAndInit() << "\n";
}


void cElHJaArrangt::DupStateFac()
{
     mStats.push_back(mStats.back());
     for (tItFac itF = mFacettes.begin() ; itF!=mFacettes.end() ; itF++)
         (*itF)->DupState();
}

void cElHJaArrangt::PopStateFac()
{
     mStats.pop_back();
     for (tItFac itF = mFacettes.begin() ; itF!=mFacettes.end() ; itF++)
         (*itF)->PopState();
}

void cElHJaArrangt::ShowStateFacette()
{
     for (tItFac itF = mFacettes.begin() ; itF!=mFacettes.end() ; itF++)
         (*itF)->ShowState();
}

void cElHJaArrangt::SetAllFacSureIfPossible()
{
     for (tItFac itF = mFacettes.begin() ; itF!=mFacettes.end() ; itF++)
         (*itF)->SetSureIfPossible();
}



Pt3dr PFromSeg(const SegComp & aSeg,REAL x,REAL y)
{
    Pt2dr aP = aSeg.from_rep_loc(Pt2dr(x,y));
    return Pt3dr(aP.x,aP.y,y);
}

void cElHJaArrangt::TestPolygoneSimple
     (
          const std::vector<Pt2dr> & aPolyg,
          const std::vector<int> &   aVSelect
     )
{
     ReInit(aPolyg);
     INT aNb = (int) aPolyg.size();
     for (INT aK=0 ; aK<aNb ; aK++)
     {
         if (aVSelect[aK])
         {
             SegComp aSeg(aPolyg[aK],aPolyg[(aK+1)%aNb]);

             cElHJaPlan3D * aPl = AddPlan
                                  (
	                               cElPlan3D
	                               (
                                          PFromSeg(aSeg,0,0),
                                          PFromSeg(aSeg,1,0),
                                          PFromSeg(aSeg,1,1)
	                               )
                                  );
            aPl->SetSegOblig(aSeg);
         }
     }
}

void cElHJaArrangt::Show()
{
     for (tItPl itPl = mPlans.begin() ; itPl!=mPlans.end() ; itPl++)
     {
         (*itPl)->Show(mVisu->WG(),255,true,false);
         Video_Win aW = *mVisu->WinOfPl((*itPl)->Num());
         (*itPl)->Show(aW,225,true,true);
     }
}


void cElHJaArrangt::TestInteractif()
{
   ELISE_ASSERT(mVisu,"cElHJaArrangt::TestInteractif");
   std::vector<Pt2dr>  aPolyg;
   std::vector<int>   aVSelec;

   mVisu->GetPolyg(aPolyg,aVSelec);


   /*
   aPolyg.clear();
   aPolyg.push_back(Pt2dr(10,10));
   aPolyg.push_back(Pt2dr(30,10));
   aPolyg.push_back(Pt2dr(30,20));
   aPolyg.push_back(Pt2dr(50,20));
   aPolyg.push_back(Pt2dr(50,50));
   aPolyg.push_back(Pt2dr(10,50));

   aVSelec.clear();
   aVSelec.push_back(1);
   aVSelec.push_back(1);
   aVSelec.push_back(1);
   aVSelec.push_back(1);
   aVSelec.push_back(1);
   aVSelec.push_back(1);
   */

   TestPolygoneSimple(aPolyg,aVSelec);
   ConstruireAll();
   Show();

/*
   // Test de recouvrement
   for (INT aK=0; true; aK++)
   {
       Show();
       cElHJaFacette * aF1 = mVisu->GetFacette(mFacettes);
       // aF1->Show(0.0,P8COL::red,false);
       aF1->ShowGlob();
       cElHJaFacette * aF2 = mVisu->GetFacette(mFacettes);
       aF2->Show(1.0,P8COL::blue,false);
       aF1->IsRecouvrt(aF2,true); 
   }
   // Visu des relation d'adjacence avec incomp
   for (INT aK=0; true; aK++)
   {
       cElHJaFacette * aF = mVisu->GetFacette(mFacettes);
       Show();
       aF->ShowGlob();
       for (tItPl itPl = mPlans.begin() ; itPl!=mPlans.end() ; itPl++)
           aF->ShowCont(P8COL::green,*(*itPl)->W());
   }
*/

/*
   ShowStateFacette();
   ShowStatistique();
   for (INT aK=0; true; aK++)
   {
       cElHJaFacette * aF = mVisu->GetFacette(mFacettes);
       Show();
       tBufFacette aBuf;
       aF->SetFacetteSure(aBuf);
       aF->PropageIncompVert(aBuf);
       cElHJaFacette::PropageVoisRecurs(aBuf);

       ShowStatistique();
       SetAllFacSureIfPossible();
       ShowStateFacette();
       ShowStatistique();

       cout << "Residu " <<  SurfResiduelle() << "\n";
   }
   */

   /*
   while (1)
   {
        std::vector<int> aVF;
        cin >> aVF ;
        getchar();
        cout << aVF << "\n";
        INT aFlg=0;
        for (INT aK=0; aK<INT(aVF.size()) ; aK++)
            aFlg |= (1<<aVF[aK]);
	// cin >> aFlg;
        StdSolFlag(aFlg,false);
   }
   */
   for (INT aFlg = 1; aFlg<(1<<mNbPl) ; aFlg++)
   {
        StdSolFlag(aFlg);
   }

}	     

void cElHJaArrangt::MakeInertieFacettes(Im2D_INT2 aMnt,INT aValForb)
{
   for (tItFac itF=mFacettes.begin() ; itF!=mFacettes.end() ; itF++)
       (*itF)->MakeInertie(aMnt,aValForb,0);
     
}

void cElHJaArrangt::ForcageSup()
{
   for (tItFac itF=mFacettes.begin() ; itF!=mFacettes.end() ; itF++)
   {
        if ((*itF)->IsIndeterminee())
        {
           (*itF)->SetFacetteSure(0);
           (*itF)->PropageIncompEnDessous(0);
        }
   }
}

bool cElHJaArrangt::GetTheSolution(bool WithForcageSup,tBufFacette & aBuf)
{
     bool anIncoh = cElHJaFacette::PropageVoisRecurs(&aBuf);
     if (anIncoh)
        return false;

     SetAllFacSureIfPossible();
     REAL aSInd = ElAbs(mStats.back().mSurfIndet);
     REAL aResidu = ElAbs(mStats.back().mSurfSure-mSurfEmpr);
     if ((aSInd<ElHJAEpsilon)&&(aResidu<ElHJAEpsilon))
        return true;

     if (! WithForcageSup)
        return false;

     ForcageSup();
     aSInd = ElAbs(mStats.back().mSurfIndet);
     aResidu = ElAbs(mStats.back().mSurfSure-mSurfEmpr);

     return ((aSInd<ElHJAEpsilon)&&(aResidu<ElHJAEpsilon));
}

void cElHJaArrangt::InitFlag(INT aFlag,tBufFacette & aBuf)
{
     mStats.back() = cStatistique(0.0,0.0,mSurfEmpr*mNbPl);
     for (tItFac itF = mFacettes.begin() ; itF!=mFacettes.end() ; itF++)
         (*itF)->SetTopState(FBool::MayBe);

     for (INT aK=0 ; aK<mNbPl ; aK++)
     {
          if ( ! (aFlag & (1<<aK)))
             mPlans[aK]->SetStatePlanInterdit();
     }

     for (INT aK=0 ; aK<mNbPl ; aK++)
     {
          if (aFlag & (1<<aK))
             mPlans[aK]->SetStatePlanWithSegOblig(&aBuf);
     }
}

bool cElHJaArrangt::StdSolFlag(INT aFlag)
{
     ELISE_ASSERT((aFlag>0) && (aFlag <(1<<mNbPl)),"cElHJaArrangt::StdSolFlag");

     tBufFacette aBuf;
     InitFlag(aFlag,aBuf);
     bool Ok = GetTheSolution(true,aBuf);
/*
     if (OK)
     {
             cElHJSol3D * aSol =  MakeSol(false);
             aSol->MakeImages(false);
              Im2D_U_INT1 aShade = aSol->ImShade();
              Im2D_INT1 aLab = aSol->ImLabel();
              ELISE_COPY
              (
	           aShade.all_pts(),
	           its_to_rgb
	           (
	              Virgule
	              (
		          aShade.in(),
		          aLab.in() * 27.9,
		          (aLab.in() >=0) * 200
	              )
	           ),
		   mVisu->WG3().orgb()  
             );
             delete aSol;
     }
*/
     // PopStateFac();
     return Ok;
}

RMat_Inertie cElHJaArrangt::MInertCurSol()
{
   RMat_Inertie aRes;
   for (tItFac anItF=mFacettes.begin() ; anItF!=mFacettes.end() ;  anItF++)
   {
       if ((*anItF)->IsSure())
       {
          aRes+= (*anItF)->MatInert();
          // cout << aRes.s() << " " << aRes.s1() << " " << aRes.s2() << "\n";
       }
   }

   return aRes;
}


cTrapuBat  cElHJaArrangt::MakeSolTrapu
           (
	         Pt2dr aDec,
		 eModeEvalZ aModeZ,
		 INT aValForbid,
		 Im2D_INT2 aMnt,
		 bool WithIndet
            )
{
   cTrapuBat aSol;
   for (INT aKPl=0 ; aKPl<mNbPl; aKPl++)
   {
      std::vector<std::vector<Pt3dr> > aVFaces  = mPlans[aKPl]->FacesSols(WithIndet);
      for (INT aKF=0 ; aKF<INT(aVFaces.size()) ; aKF++)
      {
	   aSol.AddFace(aVFaces[aKF],ElHJAEpsilon,aKPl);
      }
   }
   mSysResolv = 0;
   RMat_Inertie aMat = MInertCurSol();

   switch (aModeZ)
   {
	   case eNoCorrecZ : { } break;

	   case eCorrelEvalZ : { } break;

	   case eL2EvalZ :
           {
		   mSysResolv = &mL2SysResolv;
           }
           break;

	   case eL1EvalZ :
           {
		   mL1SysResolv.SetNbEquation(round_ni(aMat.S0()));
		   mSysResolv = &mL1SysResolv;
           }
           break;
   };
   if (aModeZ != eNoCorrecZ)
       ELISE_ASSERT(!WithIndet,"cElHJaArrangt::MakeSolTrapu");

   REAL aA,aB;
   if (mSysResolv)
   {
       mSysResolv->GSSR_Reset(false);
       for (tItFac itF=mFacettes.begin() ; itF!=mFacettes.end() ; itF++)
           if ((*itF)->IsSure())
               (*itF)->MakeInertie(aMnt,aValForbid,mSysResolv);
       mSysResolv->GSSR_SolveEqFitDroite(aA,aB);
   }


   std::vector<Pt3dr> & aVSoms = aSol.Soms();

   aVSoms.push_back(aSol.P0());
   aVSoms.push_back(aSol.P1());
   for (INT aK=0 ; aK<INT(aVSoms.size()) ; aK++)
   {
        if (aModeZ==eCorrelEvalZ)
           aVSoms[aK].z = aMat.V2toV1(aVSoms[aK].z);
	else if (mSysResolv)
	{
		aVSoms[aK].z = aB + aA *aVSoms[aK].z;
	}

        aVSoms[aK].x += aDec.x;
        aVSoms[aK].y += aDec.y;
   }
   aSol.P1() = aVSoms.back();   aVSoms.pop_back();
   aSol.P0() = aVSoms.back();   aVSoms.pop_back();

   return aSol;
}


void cElHJaArrangt::AddStatistique(REAL aSurf,const FBool & anEtat)
{
   if (anEtat== FBool::True)
   {
      mStats.back().mSurfSure += aSurf;
   }
   else if (anEtat== FBool::False)
   {
      mStats.back().mSurfImpossible += aSurf;
   }
   else if (anEtat== FBool::MayBe)
   {
      mStats.back().mSurfIndet += aSurf;
   }
}

REAL cElHJaArrangt::SurfResiduelle()
{
    return ElAbs(     mStats.back().mSurfSure
		    + mStats.back().mSurfIndet
		    -mSurfEmpr
		);
}

void cElHJaArrangt::ShowStatistique()
{
     cStatistique aSt = mStats.back();
     REAL aSSure = aSt.mSurfSure;
     REAL aSImp = aSt.mSurfImpossible;
     REAL aSInd = aSt.mSurfIndet;
     cout << "Invar " << (aSSure+aSImp+aSInd)-mSurfEmpr*mNbPl 
	  << " Residu " <<  (aSSure+aSInd-mSurfEmpr)
	  << "\n";
}

void cElHJaArrangt::TriTopologiqueFacette()
{
   tContFac aNewTabF;

   for (tItFac anItF=mFacettes.begin() ; anItF!=mFacettes.end() ;  anItF++)
   {
      if ((*anItF)->NbThisEnDessous() ==0)
         aNewTabF.push_back(*anItF);
   }

   INT aK= 0;
   while (aK!=INT( aNewTabF.size()))
   {
       aNewTabF[aK]->AddDessouSansDessus(aNewTabF);
       aK++;
   }

   ELISE_ASSERT
   (
       mFacettes.size() == aNewTabF.size(),
       "cElHJaArrangt::TriTopologiqueFacette"
   );
   mFacettes =  aNewTabF;
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
