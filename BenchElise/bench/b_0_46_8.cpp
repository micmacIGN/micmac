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


#include "StdAfx.h"
#include "bench.h"

class cBenchAppuiGridVue;
class cChantierBAG;

   //**********************************************************

class cBenchAppuiGridVue
{
      public :
         cBenchAppuiGridVue
         (
              cSetEqFormelles & aSet,
              cTriangulFormelle & aTri,
              REAL a01,REAL a02,REAL a12,Pt3dr aCOpt,
              REAL aRanTeta, REAL aRanP
         );

         Pt3dr M2C(Pt3dr aP);
         void AddObsAppui(Pt3dr aPM,Pt2dr aPIm);
	 REAL AddEqAppuis(cChantierBAG &);
	 void ShowResidRot();

      private :


    // Dans le sens Cam->Monde
         
         ElRotation3D         mRotReelle;
         ElRotation3D         mRotEstimee;
         cRotationFormelle  * pRotF;
         cAppuiGridEq       * pEqAp;
         std::list<Appar23>   mListApp;
         
};


class cChantierBAG
{
    public :
         cChantierBAG(REAL Ouv,INT aNbTri);
	 void AddVue( REAL a01,REAL a02,REAL a12,Pt3dr aCOpt);
	 void CloseVues();

          void SetRanTeta(REAL);
          void SetRanP(REAL);
	  void AddEqAppuis();


          void AddPlanObsAppuis(Box2dr aBox,REAL aZ,REAL aStep);
          Pt2dr ToIm(cBenchAppuiGridVue & aCam, Pt3dr aPM);

    private :

          void AddAnObsAppui(cBenchAppuiGridVue &, Pt3dr aP);
          void AddAnObsAppui2AllVues(Pt3dr aP);
          Pt2dr Proj2Im(Pt2dr aP);
          REAL EcartDist();


          cSetEqFormelles            mSetEqF;
          REAL                       mOuv;
          Pt2dr                      mSzOuv;
          Box2dr                     mBoxOuv;
          cTriangulFormelle *        mTriF;
	  REAL                       mRanTeta;
	  REAL                       mRanCopt;
	  // Se sont focale et PP qui joue le role de la dist
	  REAL                       mF;
	  Pt2dr                      mPP;
	  std::list<cBenchAppuiGridVue *> mLVues;
};


/*******************************************************/
/*                                                     */
/*             cBenchAppuiGridVue                      */
/*                                                     */
/*******************************************************/

cBenchAppuiGridVue::cBenchAppuiGridVue
(
      cSetEqFormelles & aSet,
      cTriangulFormelle & aTri,
      REAL a01,REAL a02,REAL a12,
      Pt3dr aCOpt,
      REAL aRanTeta,REAL aRanP
)  :
       mRotReelle  (ElRotation3D(aCOpt,a01,a02,a12)),
       mRotEstimee (ElRotation3D
                       (
                          aCOpt+Pt3dr(NRrandC(),NRrandC(),NRrandC())*aRanP,
                          a01+NRrandC()*aRanTeta,
                          a02+NRrandC()*aRanTeta,
                          a12+NRrandC()*aRanTeta
                       )
                   ),
       pRotF       (aSet.NewRotation
                         (
                              cNameSpaceEqF::eRotLibre,
                              mRotEstimee,
                              (cRotationFormelle *) 0,
                              ""
                         )
                   )        ,
        pEqAp      (aSet.NewEqAppuiGrid(aTri,*pRotF)) 
{
}

Pt3dr cBenchAppuiGridVue::M2C(Pt3dr aP)
{
   //return mRotReelle.IRecAff(aP);
   return mRotReelle.ImRecAff(aP); // __NEW
}


void cBenchAppuiGridVue::AddObsAppui(Pt3dr aPM,Pt2dr aPIm)
{
    mListApp.push_back(Appar23(aPIm,aPM));
}


REAL cBenchAppuiGridVue::AddEqAppuis(cChantierBAG & aCH)
{
  REAL D = 0.0;
  INT  CPT = 0;
  for 
  (
      std::list<Appar23>::iterator itAp = mListApp.begin();
      itAp != mListApp.end();
      itAp++
  )
  {
    Pt2dr aP = pEqAp->AddAppui(itAp->pter,itAp->pim,1.0);
    D += euclid(aP);
    CPT++;
  }
  return D/CPT;
}


/*******************************************************/
/*                                                     */
/*             cChantierBAG                            */
/*                                                     */
/*******************************************************/


cChantierBAG::cChantierBAG(REAL anOuv,INT aNbTri) :
     mSetEqF (),
     mOuv    (anOuv),
     mSzOuv  (Pt2dr(mOuv,mOuv)),
     mBoxOuv (-mSzOuv,mSzOuv),
     //mTriF   (mSetEqF.NewTriangulFormelle(mBoxOuv,aNbTri,1.8/aNbTri)),
     mTriF   (mSetEqF.NewTriangulFormelle(2,mBoxOuv,aNbTri,1.8/aNbTri)), // __NEW
     mRanTeta (0.05),
     mRanCopt (0.5),
     mF      (1.05),
     mPP     (0,0)
{
}

Pt2dr cChantierBAG::Proj2Im(Pt2dr aP)
{
    return (mPP+aP)/mF;
}

void cChantierBAG::AddVue(REAL a01,REAL a02,REAL a12,Pt3dr aCOpt)
{
     mLVues.push_back
     (
         new cBenchAppuiGridVue
	     (
	        mSetEqF,*mTriF,
		a01,a02,a12,aCOpt,
                mRanTeta,mRanCopt
	     )
     );
}

Pt2dr cChantierBAG::ToIm(cBenchAppuiGridVue & aCam, Pt3dr aPM)
{
      Pt3dr aPC = aCam.M2C(aPM);
      ELISE_ASSERT(aPC.z !=0,"cChantierBAG::AddAnAppui");

      Pt2dr aPProj (aPC.x/aPC.z,aPC.y/aPC.z);
      return  Proj2Im(aPProj);
}

REAL cChantierBAG::EcartDist()
{
     REAL aD = 0;
     INT aCPt = 0;
     INT aNB = 20;
     for (INT aKX=0 ; aKX <= aNB ; aKX++)
         for (INT aKY=0 ; aKY <= aNB ; aKY++)
	 {
             Pt2dr aP = mBoxOuv.FromCoordBar(Pt2dr(aKX/REAL(aNB),aKY/REAL(aNB)));
	     Pt2dr aQ1 = mTriF->Direct(aP);
	     Pt2dr aQ2 = aP* mF;
	     aD += euclid(aQ1,aQ2);
	     aCPt++;
	 }
     return aD / aCPt;
}


void cChantierBAG::AddAnObsAppui(cBenchAppuiGridVue & aCam, Pt3dr aPM)
{
      Pt2dr aPIm = ToIm(aCam,aPM);

      if (mBoxOuv.inside(aPIm))
         aCam.AddObsAppui(aPM,aPIm);
}

void cChantierBAG::AddAnObsAppui2AllVues(Pt3dr aPM)
{
   for 
   (
      std::list<cBenchAppuiGridVue *>::iterator itV = mLVues.begin();
      itV != mLVues.end();
      itV++
   )
      AddAnObsAppui(**itV,aPM);
}

void cChantierBAG::AddPlanObsAppuis(Box2dr aBox,REAL aZ,REAL aStep)
{
     for (REAL anX = aBox._p0.x; anX<aBox._p1.x ; anX+=aStep)
        for (REAL anY = aBox._p0.y; anY<aBox._p1.y ; anY+=aStep)
	{
		AddAnObsAppui2AllVues(Pt3dr(anX,anY,aZ));
	}
}

void cChantierBAG::AddEqAppuis()
{
   //mSetEqF.AddContrainte(mTriF->ContraintesRot());
   mSetEqF.AddContrainte(mTriF->ContraintesRot(),true/*isStrict*/); // __NEW
   for 
   (
      std::list<cBenchAppuiGridVue *>::iterator itV = mLVues.begin();
      itV != mLVues.end();
      itV++
   )
   {
      REAL D = (*itV)->AddEqAppuis(*this);
      cout << "Residu : " << D << "\n";
      cout << "Ec/Dist "  << EcartDist() << "\n";
      (*itV)->ShowResidRot();
   }
   mSetEqF.SolveResetUpdate();
}

void cChantierBAG::CloseVues()
{
     mSetEqF.SetClosed();
}

void cBenchAppuiGridVue::ShowResidRot()
{
    ElRotation3D aR = pRotF->CurRot();
    cout << "Ecart/Rot " 
         << euclid(mRotReelle.tr()-aR.tr()) << " " 
         << mRotReelle.Mat().L2(aR.Mat()) << "\n";
}

/*******************************************************/
/*                                                     */
/*             ::                                      */
/*                                                     */
/*******************************************************/

void BenchEqAppGrid()
{
    cChantierBAG aCH(0.5,4);
    aCH.AddVue(0.1,0.1,0.1,Pt3dr(1,1,1));

    aCH.CloseVues();
    aCH.AddPlanObsAppuis(Box2dr(Pt2dr(-10,-10),Pt2dr(10,10)),10,1.0);
    aCH.AddPlanObsAppuis(Box2dr(Pt2dr(-10,-10),Pt2dr(10,10)),20,1.0);
    for (INT aK=0 ; aK<100 ; aK++)
    {
      aCH.AddEqAppuis();
    }
    getchar();
}



