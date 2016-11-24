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
#include "algo_geom/delaunay_mediatrice.h"
#include "graphes/graphe.h"
#include "graphes/graphe_implem.h"
#include "algo_geom/qdt.h"
#include "algo_geom/qdt_implem.h"
#include "graphes/algo_planarite.h"
#include "graphes/algo_pcc.h"




class cTFI_Som;
class cTriangulFormelleImplem;



typedef INT cTFI_AttrARC;
typedef ElSom<cTFI_AttrSom *,cTFI_AttrARC> tTFISom;
typedef ElArc<cTFI_AttrSom *,cTFI_AttrARC> tTFIArc;
typedef ElSubGraphe<cTFI_AttrSom *,cTFI_AttrARC> tTFISubGr;
typedef ElPcc<cTFI_AttrSom *,cTFI_AttrARC>       tTFIPCC;

template <class Type> Pt2d<Type> ToPt(const std::vector<Type> & aV)
{
    ELISE_ASSERT(aV.size()==2,"ToPt dim should be 2 !! ");
    return Pt2d<Type>(aV[0],aV[1]);
}

/*
template <class Type> const Type & ToV(const std::vector<Type> & aV)
{
	    ELISE_ASSERT(aV.size()==1,"ToPt dim should be 2 !! ");
	        return Pt2d<Type>(aV[1]);
}
*/


template <class Type> std::vector<Type> ToVec(const  Pt2d<Type> & aP)
{
   std::vector<Type> aV;
   aV.push_back(aP.x);
   aV.push_back(aP.y);
   return aP;
}





class cTFI_SubGr_Std : public tTFISubGr
{
	public :
          Pt2dr pt(tTFISom & aSom){return aSom.attr()->PosPt();}
          REAL   pds(tTFIArc & anArc) 
                 {
                    return  euclid
                            (
                                this->pt(anArc.s1()),
                                this->pt(anArc.s2())
                            );
                 }
	private:
};

class cGetSegFromTFI
{
     public :
       Seg2d operator () (tTFIArc * pArc)
       {
          return Seg2d(pArc->s1().attr()->PosPt(),pArc->s2().attr()->PosPt());
       }

};
typedef ElQT<tTFIArc *,Seg2d,cGetSegFromTFI>  tTFIQtArc;

 
class cGetTriFromTFI
{
      public :
         const cElTriangleComp & operator()(cTFI_Triangle * aTri)
         {
              return aTri->TriGeom();
         }
};

typedef ElQT<cTFI_Triangle *,cElTriangleComp,cGetTriFromTFI>  tTFIQtTri;



#define NbTol 3

class cTriangulFormelleImplem : public cTriangulFormelle
{
     public :
         virtual void SetAllPtCur(const ElDistortion22_Gen &);
         friend class cSetEqFormelles;
         cTriangulFormelleImplem
	 (
              int aDim,
              cSetEqFormelles &,
              Box2dr,
              REAL DMax
         );
	 ~cTriangulFormelleImplem();

	 tTFISom & AddSom(Pt2dr aP,const std::vector<double> & aVals);
	 void  AddArcSiOkTopo(tTFISom *,tTFISom *);

	 void Triangulate();
	 void Finish();

	 Pt2di TPI(Pt2dr aP) 
	       {
		       return Pt2di(
			       round_ni((aP.x-mP0.x)*mScale +mRabW),
			       round_ni((aP.y-mP0.y)*mScale +mRabW)
			       );
	       }
	 Pt2dr TPRI(Pt2dr aP) {return Pt2dr(TPI(aP));}

	 virtual void Show() ;
	 virtual void Show(ElPackHomologue) ;


     private  :

         cTFI_AttrSom * SomCentral() ;
         cTFI_AttrSom * VecHorz() ;

         mutable std::set<cTFI_Triangle *> mSetTri;
	 cTFI_Triangle & GetTriFromP(const Pt2dr & aP) const;
	 cTFI_Triangle * GetTriFromP(const Pt2dr & aP,REAL aDTol) const;
	 void TestOneTri();




         Pt2dr APointInTri() const;
	 void SetTolMax(REAL aTol);



	 INT                                    mNbPt;
	 Pt2dr                                  mSomPt;
	 ElGraphe<cTFI_AttrSom *,cTFI_AttrARC>  mGr;
         tTFIPCC                                mPCC;
	 tTFIQtArc *                          pQtArc;
	 tTFIQtTri *                          pQtTri;

	 std::vector<tTFISom *>       mNodes;

	 INT                       mSzInterne;
	 INT                       mRabW;
	 Video_Win *               pW;
	 Pt2dr                     mP0;
	 Pt2dr                     mP1;
	 REAL                      mScale;
	 REAL                      mDType;

	 tTFISom *                 mSomCentrale;
	 tTFISom *                 mSomVecHorz;
	 INT                       mNbArc;

	 double                    mDTol[NbTol];
	 REAL                      mDMax;
};



/************************************************************/
/*                                                          */
/*          cTFI_Triangle                                   */
/*                                                          */
/************************************************************/

cTFI_Triangle::cTFI_Triangle
(
     cTFI_AttrSom & aS1,
     cTFI_AttrSom & aS2,
     cTFI_AttrSom & aS3
) :
   mDim      (aS1.Dim()),
   mA1       (aS1),
   mIntervA1 (aS1.Interv(),"SomA1"),
   mIntervB1 (aS1.Interv(),"SomB1"),

   mA2       (aS2),
   mIntervA2 (aS2.Interv(),"SomA2"),
   mIntervB2 (aS2.Interv(),"SomB2"),

   mA3       (aS3),
   mIntervA3 (aS3.Interv(),"SomA3"),
   mIntervB3 (aS3.Interv(),"SomB3"),

   mTri (aS1.PosPt(),aS2.PosPt(),aS3.PosPt())
{
   ELISE_ASSERT(mDim==aS2.Dim(),"Dim dif in cTFI_Triangle::cTFI_Triangle");
   ELISE_ASSERT(mDim==aS3.Dim(),"Dim dif in cTFI_Triangle::cTFI_Triangle");

   ELISE_ASSERT
   (
      (&(aS1.Set())==&(aS2.Set())) &&(&(aS1.Set())==&(aS3.Set())),
      "Dif set in cTFI_Triangle"
   );

/*
   for (int aD=0 ; aD<mDim ; aD++)
   {
       mVIndexe.push_back(mIntervA1.I0()+aD);
       mVIndexe.push_back(mIntervA2.I0()+aD);
       mVIndexe.push_back(mIntervA3.I0()+aD);
   }
   mVIndexe.push_back(mIntervA1.I0());
   mVIndexe.push_back(mIntervA1.I0()+1);
   mVIndexe.push_back(mIntervA2.I0());
   mVIndexe.push_back(mIntervA2.I0()+1);
   mVIndexe.push_back(mIntervA3.I0());
   mVIndexe.push_back(mIntervA3.I0()+1);
*/
}

/*
const std::vector<int> & cTFI_Triangle::VecOfIndexe() const
{
   return mVIndexe;
}
*/

cSetEqFormelles & cTFI_Triangle::Set()
{
   return mA1.Set();
}

cTFI_Triangle  * cTFI_Triangle::NewOne
(
     cTFI_AttrSom & aS1,
     cTFI_AttrSom & aS2,
     cTFI_AttrSom & aS3
)
{
        bool Swap = cElTriangleComp::ToSwap(aS1.PosPt(),aS2.PosPt(),aS3.PosPt());
	cTFI_Triangle * aRes = Swap                           ? 
                               new cTFI_Triangle(aS1,aS3,aS2) : 
                               new cTFI_Triangle(aS1,aS2,aS3) ;
	return aRes;
}


const cElTriangleComp & cTFI_Triangle::TriGeom() const { return  mTri;}


std::vector<double> cTFI_Triangle::InterpolVals(const Pt2dr & aP) const
{
   std::vector<double> aRes;
   Pt3dr aCoord = mTri.CoordBarry(aP);
   for  (int aD=0 ; aD<mDim ; aD++)
   {
      aRes.push_back
      (
            mA1.ValsInc()[aD] * aCoord.x
         +  mA2.ValsInc()[aD] * aCoord.y
         +  mA3.ValsInc()[aD] * aCoord.z
      );
   }
   return aRes;
}
/*
*/
const std::vector<Fonc_Num> &  cTFI_Triangle::Inc1() const {return mA1.Incs();}
const std::vector<Fonc_Num> &  cTFI_Triangle::Inc2() const {return mA2.Incs();}
const std::vector<Fonc_Num> &  cTFI_Triangle::Inc3() const {return mA3.Incs();}



const cIncIntervale &  cTFI_Triangle::IntervA1 () const {return mIntervA1;}
const cIncIntervale &  cTFI_Triangle::IntervA2 () const {return mIntervA2;}
const cIncIntervale &  cTFI_Triangle::IntervA3 () const {return mIntervA3;}

const cIncIntervale &  cTFI_Triangle::IntervB1 () const {return mIntervB1;}
const cIncIntervale &  cTFI_Triangle::IntervB2 () const {return mIntervB2;}
const cIncIntervale &  cTFI_Triangle::IntervB3 () const {return mIntervB3;}


int cTFI_Triangle::Dim() const
{
   return mDim;
}


cElPlan3D cTFI_Triangle::CalcPlancCurValAsZ() const
{
   return cElPlan3D(mA1.P3ValAsZ(),mA2.P3ValAsZ(),mA3.P3ValAsZ());
}

cTFI_AttrSom & cTFI_Triangle::S1() { return mA1;}
cTFI_AttrSom & cTFI_Triangle::S2() { return mA2;}
cTFI_AttrSom & cTFI_Triangle::S3() { return mA3;}

/************************************************************/
/*                                                          */
/*          cTFI_AttrSom                                    */
/*                                                          */
/************************************************************/


cTFI_AttrSom::cTFI_AttrSom
(
    cSetEqFormelles &            aSet,
    Pt2dr                        aPos,
    const std::vector<double> &  aVInit
) :
    mSet     (aSet),
    mInterv  (false,std::string("Pts"),aSet),
    mPos     (aPos),
    mValsInc (aVInit),
    mNumInc  (aSet.Alloc().CurInc()),
    mDim     ((int)mValsInc.size())
{
  for (int aK=0 ; aK<mDim ; aK++)
      mIncs.push_back(aSet.Alloc().NewF("cTFI_AttrSom","P"+ToString(aK),&(mValsInc[aK])));
   mInterv.Close();
}

int cTFI_AttrSom::Dim() const
{
   return mDim;
}

cSetEqFormelles & cTFI_AttrSom::Set()
{
   return mSet;
}

/*
cTFI_AttrSom * cTFI_AttrSom::Som2D(cSetEqFormelles & aSet,Pt2dr aPos,Pt2dr aVal)
{
   return new cTFI_AttrSom(aSet,aPos,ToVec(aVal));
}
*/

void cTFI_AttrSom::AssertD2() const { ELISE_ASSERT( mDim==2, "cTFI_AttrSom::AssertD2"); }
void cTFI_AttrSom::AssertD1() const { ELISE_ASSERT( mDim==1, "cTFI_AttrSom::AssertD1"); }


const cIncIntervale &   cTFI_AttrSom::Interv() const { return mInterv; }
const Pt2dr &  cTFI_AttrSom::PosPt() const { return mPos; }
const std::vector<Fonc_Num> & cTFI_AttrSom::Incs() const { return mIncs; }
const std::vector<double> & cTFI_AttrSom::ValsInc() const { return mValsInc; }


Pt3dr  cTFI_AttrSom::P3ValAsZ() const
{
   AssertD1();
   return  Pt3dr(mPos.x,mPos.y,mValsInc[0]);
}

    // Specific D2

void cTFI_AttrSom::SetValCurPt(cSetEqFormelles & aSet,const Pt2dr & aP)
{
    AssertD2();
    aSet.Alloc().SetVar(aP.x,mNumInc);
    aSet.Alloc().SetVar(aP.y,mNumInc+1);
}
Pt2dr cTFI_AttrSom::ValCurAsPt()  const 
{
   AssertD2();
   return Pt2dr(mValsInc[0],mValsInc[1]);
}
Pt2d<Fonc_Num> cTFI_AttrSom::ValsIncAsPt()  const 
{
    AssertD2();
   return Pt2d<Fonc_Num>(mIncs[0],mIncs[1]);
}

Fonc_Num   cTFI_AttrSom::ValsIncAsScal() const
{
    AssertD1();
    return mIncs[0];
}


/************************************************************/
/*                                                          */
/*    cTriangulFormelle                                     */
/*                                                          */
/************************************************************/

cTriangulFormelle::~cTriangulFormelle()
{
}


void cTriangulFormelle::AssertD2() const
{
    ELISE_ASSERT(mDim==2,"cTriangulFormelle::AssertD2");
}

std::vector<double> cTriangulFormelle::ValsOfPt(const Pt2dr &  aP) const
{
    const cTFI_Triangle & aTri = GetTriFromP(aP);

    return aTri.InterpolVals(aP);
}

Pt2dr cTriangulFormelle::Direct(Pt2dr  aP) const
{
    AssertD2();
    return ToPt(ValsOfPt(aP));
}

void  cTriangulFormelle::Diff(ElMatrix<REAL> & aMat,Pt2dr aP) const
{
     AssertD2();
    // Pas Tres Optimise mais bon ....
    const cTFI_Triangle & aTri = GetTriFromP(aP);

    Pt2dr aP0 = ToPt(aTri.InterpolVals(aP));
    Pt2dr aGx = ToPt(aTri.InterpolVals(aP+Pt2dr(1,0)))-aP0;
    Pt2dr aGy = ToPt(aTri.InterpolVals(aP+Pt2dr(0,1)))-aP0;

    aMat.set_to_size(2,2);

    SetCol(aMat,0,aGx);
    SetCol(aMat,1,aGy);

}

cTriangulFormelle::cTriangulFormelle(int aDim,cSetEqFormelles & aSet) :
    cElemEqFormelle (aSet,false),
    mSet            (aSet),
    mDim            (aDim),
    mTrianFTol      (cContrainteEQF::theContrStricte)
{
}

cSetEqFormelles & cTriangulFormelle::Set()
{
   return mSet;
}


    // tContFcteur      ContraintesRot() ;
cMultiContEQF     cTriangulFormelle::ContraintesAll() 
{
     cMultiContEQF aRes;
     AddFoncRappInit(aRes,0,mNumIncN-mNumInc0,mTrianFTol);
     return aRes;
}
    
cMultiContEQF cTriangulFormelle::ContraintesRot() 
{
   cTFI_AttrSom * aSC =  SomCentral();
   const cIncIntervale & aIC = aSC->Interv();
   cMultiContEQF aRes;
   AddFoncRappInit(aRes,aIC.I0Alloc()-mNumInc0,aIC.I1Alloc()-mNumInc0,mTrianFTol);


   cTFI_AttrSom * aSH =  VecHorz();
   const cIncIntervale & aIH = aSH->Interv();
   AddFoncRappInit(aRes,aIH.I0Alloc()+1-mNumInc0,aIH.I1Alloc()-mNumInc0,mTrianFTol);


   return aRes;
}
    
const std::vector<cTFI_Triangle *>&  cTriangulFormelle::VTri() const
{
   return mVTri;
}

/************************************************************/
/*                                                          */
/*    cTriangulFormelleImplem                               */
/*                                                          */
/************************************************************/

cTriangulFormelleImplem::cTriangulFormelleImplem
(
       int               aDim,
       cSetEqFormelles & aSet,
       Box2dr            aBox,
       REAL              aDMax
) :
  cTriangulFormelle  (aDim,aSet),
  mNbPt              (0),
  mSomPt             (0.0,0.0),
  pQtArc             (0),
  pQtTri             (0),
  mSzInterne         (700),
  mRabW              (10),
  pW                 (0),
                     /*(Video_Win::PtrWStd
		           (Pt2di(mSzInterne+2*mRabW,mSzInterne+2*mRabW))
                     ),*/
  mP0                (aBox._p0),
  mP1                (aBox._p1),
  mScale             (mSzInterne/dist8(mP1-mP0)),
  mSomCentrale       (0),
  mSomVecHorz        (0),
  mNbArc             (0),
  mDMax              (aDMax)
{
}

void cTriangulFormelleImplem::SetAllPtCur(const ElDistortion22_Gen & aDist)
{
    AssertD2();
    for (int aK=0 ; aK<int(mNodes.size()) ; aK++)
    {
        cTFI_AttrSom * aN = mNodes[aK]->attr();
        aN->SetValCurPt(mSet,aDist.Direct(aN->PosPt()));
    }
}


tTFISom &  cTriangulFormelleImplem::AddSom(Pt2dr aPos,const std::vector<double> & aVals)
{
     ELISE_ASSERT
     (
       pQtArc==0,
       "Add Som After Arc in cTriangulFormelleImplem"
     );

     ELISE_ASSERT
     (
         int(aVals.size())==mDim,
	 "Incorent dims in cTriangulFormelleImplem::AddSom"
     );

     mSomPt += aPos;

     tTFISom & aSom = mGr.new_som(new cTFI_AttrSom(Set(),aPos,aVals));
     mNodes.push_back(&(aSom));
     if (pW)
        pW->draw_circle_loc(Pt2dr(TPI(aSom.attr()->PosPt())),3.0,pW->pdisc()(P8COL::blue));

     mNbPt++;
     return aSom;
}

void  cTriangulFormelleImplem::AddArcSiOkTopo(tTFISom * aS1,tTFISom * aS2)
{
    if (aS1== aS2)
       return;

    if (pQtArc==0)
    {
        pQtArc = new tTFIQtArc
		     (
			cGetSegFromTFI(),
			Box2dr(mP0 -Pt2dr(2,2),mP1+Pt2dr(2,2)),
			10,
                        mDType / (4.0 * sqrt(1.0+mNodes.size()))
                     );
    }
    if (mGr.arc_s1s2(*aS1,*aS2))
       return;

    std::set<tTFIArc*> aSet;
    SegComp  aSeg(aS1->attr()->PosPt(),aS2->attr()->PosPt());
    REAL aDSeg = aSeg.length();
    pQtArc->RVoisins(aSet,aSeg,aDSeg/50.0);

    for 
    (
       std::set<tTFIArc*>::const_iterator iTA = aSet.begin();
       iTA != aSet.end();
       iTA++
    )
    {
        tTFIArc & anArc = ** iTA;
	if ((! anArc.IsAdj(*aS1)) && (! anArc.IsAdj(*aS2)))
	   return;
    }

    cTFI_SubGr_Std aSubGr;
    if (mPCC.inf_dist(*aS1,*aS2,aSubGr,1.01*aDSeg,eModePCC_Somme))
       return;


    tTFIArc & anArc = mGr.add_arc(*aS1,*aS2,0,0);
    pQtArc->insert(&anArc);
    if (pW)
    {			
       pW->draw_seg
       (
             Pt2dr(TPI(aS1->attr()->PosPt())),
             Pt2dr(TPI(aS2->attr()->PosPt())),
             pW->pdisc()(P8COL::red)
       );
    }
    mNbArc++;
}

cTriangulFormelleImplem::~cTriangulFormelleImplem()
{
    delete pQtArc;
    delete pQtTri;
}


void cTriangulFormelleImplem::TestOneTri()
{
     if (pW)
     {
	Pt2dr aP =  pW->clik_in()._pt;
	pW->draw_circle_loc(Pt2dr(TPI(aP)),3.0,pW->pdisc()(P8COL::yellow));

	cElTriangleComp  aTr = GetTriFromP(aP).TriGeom();

	pW->draw_seg(TPRI(aTr.P0()),TPRI(aTr.P1()),pW->pdisc()(P8COL::yellow));
	pW->draw_seg(TPRI(aTr.P1()),TPRI(aTr.P2()),pW->pdisc()(P8COL::yellow));
	pW->draw_seg(TPRI(aTr.P2()),TPRI(aTr.P0()),pW->pdisc()(P8COL::yellow));


	pW->draw_circle_loc(APointInTri(),6.0,pW->pdisc()(P8COL::cyan));
     }
}
Pt2dr cTriangulFormelleImplem::APointInTri() const
{
   ELISE_ASSERT(mNbPt >=3,"cTriangulFormelleImplem::APointInTri");
   return mSomPt / REAL(mNbPt);
}



cTFI_Triangle * cTriangulFormelleImplem::GetTriFromP (const Pt2dr & aP,REAL aDTol) const
{
    mSetTri.clear();
    pQtTri->RVoisins(mSetTri,aP,aDTol);

    cTFI_Triangle * aRes = 0; 
    REAL aDMin =aDTol+1;

    for 
    (
        std::set<cTFI_Triangle *>::const_iterator iT = mSetTri.begin();
	iT != mSetTri.end();
	iT++
    )
    {
        REAL  aD = sqrt((*iT)->TriGeom().square_dist(aP));
	if (aD < aDMin)
	{
            aDMin = aD;
	    aRes = *iT;
	}
    }
    return aRes;


}

cTFI_Triangle & cTriangulFormelleImplem::GetTriFromP (const Pt2dr & aP) const
{
    for (INT aK=0; aK<NbTol ; aK++)
    {
        cTFI_Triangle * aRes = GetTriFromP (aP,mDTol[aK]);
	if (aRes != 0)
           return *aRes;
    }
    ELISE_ASSERT(false,"cTriangulFormelleImplem::GetTriFromP");
    cTFI_Triangle * aRes = 0;
    return *aRes;
}

void cTriangulFormelleImplem::SetTolMax(REAL aTol)
{
      ELISE_ASSERT(NbTol==3,"NbTol!=3 in cTriangulFormelleImplem::Triangulate");
      mDTol[0] = ElMin(aTol,1e-4)*mDType;
      mDTol[1] = ElMin(aTol,1e-1)*mDType;
      mDTol[2] = aTol*mDType;
}

    //=================================================================
    //
    //    Classe support a la triangulation
    //    cTFI_Fpt
    //    cTFIDelauAct
    //
    //=================================================================

class cTFI_Fpt
{
    public :
        Pt2di operator ()(tTFISom * aPtr) 
	{
            return mTFI.TPI(aPtr->attr()->PosPt());
	}

	cTFI_Fpt (cTriangulFormelleImplem & aTFI) : mTFI(aTFI) {}
	cTriangulFormelleImplem & mTFI;

};

class cTFIDelauAct
{
    public :
       cTFIDelauAct(cTriangulFormelleImplem & aTri) : mTri(aTri) {}
       void operator () (tTFISom * aS1,tTFISom * aS2,bool isDeg)
       {
            mTri.AddArcSiOkTopo(aS1,aS2);
       }
    private :
       cTriangulFormelleImplem & mTri;
};

void cTriangulFormelleImplem::Show() 
{
   AssertD2();
   if (pW)
   {
      pW->clear();
      for (INT aK=0 ; aK <INT(mNodes.size()) ; aK++)
      {
          cTFI_AttrSom * aSom = mNodes[aK]->attr();
	  pW->draw_circle_loc(TPRI(aSom->PosPt()),3.0,pW->pdisc()(P8COL::white));
	  pW->draw_seg(TPRI(aSom->PosPt()),TPRI(aSom->ValCurAsPt()),pW->pdisc()(P8COL::red));
      }
   }
}
void cTriangulFormelleImplem::Show(ElPackHomologue aPack) 
{
   if (pW)
   {
      for (ElPackHomologue::iterator iT=aPack.begin(); iT!=aPack.end() ; iT++)
      {
	  pW->draw_circle_loc(TPRI(iT->P1()),1.0,pW->pdisc()(P8COL::green));
	  pW->draw_circle_loc(TPRI(iT->P2()),1.0,pW->pdisc()(P8COL::green));
      }
   }
}


class CplSom
{
     public :
        CplSom(tTFISom * aS1,tTFISom * aS2,Pt2dr Center) :
		mS1 (aS1),
		mS2 (aS2)
		
	{
		Pt2dr aP1 = aS1->attr()->PosPt();
		Pt2dr aP2 = aS2->attr()->PosPt();

		REAL d12 = euclid(aP1,aP2);
		REAL d1C = euclid(aP1,Center);
		REAL d2C = euclid(aP2,Center);

		mCost = d12 + ElAbs(d1C-d2C) / 10.0;
	}


        tTFISom * mS1;
        tTFISom * mS2;
	REAL      mCost;
};

bool operator < (const CplSom & S1,const CplSom & S2)
{
    return  S1.mCost < S2.mCost;
}

void cTriangulFormelleImplem::Triangulate()
{
      mVTri.clear();
      mDType = euclid(mP1,mP0);

      SetTolMax(1e2);
      Pt2dr aCenter = (mP1+mP0)/2.0;

      {
          INT NbS = (INT) mNodes.size();
          std::vector<CplSom> aVCpl;

          REAL aDSeuil = mDMax * dist8(mP0-mP1);
          for (INT aK1=0 ; aK1<NbS ; aK1++)
              for (INT aK2=aK1+1 ; aK2<NbS ; aK2++)
	      {
	          Pt2dr aP1 = mNodes[aK1]->attr()->PosPt();
	          Pt2dr aP2 = mNodes[aK2]->attr()->PosPt();
	          if (euclid(aP1,aP2) < aDSeuil)
                      aVCpl.push_back(CplSom(mNodes[aK1],mNodes[aK2],aCenter));
	  }

          std::sort(aVCpl.begin(),aVCpl.end());
          for (INT aK=0 ; aK<INT(aVCpl.size()) ; aK++)
	      AddArcSiOkTopo(aVCpl[aK].mS1,aVCpl[aK].mS2);
      }


      cTFI_SubGr_Std aSubG;
      ElPartition<tTFIArc *> aPart;
      ElFifo<Pt2dr> aFace;
      aFace.set_circ(true);

      all_face_trigo (mGr,aSubG,aPart);

      aFace.set_circ(true);
      for (INT aKF=0 ; aKF< aPart.nb() ; aKF++)
      {
          ElSubFilo<tTFIArc *> aF = aPart[aKF]; 
	  if (aF.nb() > 3)
	  {
             aFace.clear();
	     for (INT aKP=0 ; aKP<aF.nb() ; aKP++)
                aFace.pushlast(aF[aKP]->s1().attr()->PosPt());
	     if (surf_or_poly(aFace) < 0)
	     {
                 std::vector<CplSom> aVCpl;
	         for (INT aKP1 = 0 ; aKP1<aF.nb() ; aKP1++)
	             for (INT aKP2 = aKP1+2 ; aKP2<aF.nb() ; aKP2++)
                         aVCpl.push_back(CplSom(&aF[aKP1]->s1(),&aF[aKP2]->s1(),aCenter));
                std::sort(aVCpl.begin(),aVCpl.end());
                for (INT aK=0 ; aK<INT(aVCpl.size()) ; aK++)
	            AddArcSiOkTopo(aVCpl[aK].mS1,aVCpl[aK].mS2);
	     }
	  }
      }

      pQtTri = new tTFIQtTri
                   (
                      cGetTriFromTFI(),
                      Box2dr(mP0 -Pt2dr(2,2),mP1+Pt2dr(2,2)),
                      10,
                      mDType / (4.0 * sqrt(1.0+mNodes.size()))
                   );


      aPart.clear();
      all_face_trigo (mGr,aSubG,aPart);
      for (INT aKF=0 ; aKF< aPart.nb() ; aKF++)
      {
          aFace.clear();
          ElSubFilo<tTFIArc *> aF = aPart[aKF]; 
	  for (INT aKP=0 ; aKP<aF.nb() ; aKP++)
             aFace.pushlast(aF[aKP]->s1().attr()->PosPt());
	  if (aF.nb()==3 && ( surf_or_poly(aFace)))
	  {
               cTFI_Triangle * aTri = 
                         cTFI_Triangle::NewOne
		         (
			     *aF[0]->s1().attr(),
			     *aF[1]->s1().attr(),
			     *aF[2]->s1().attr()
			 );
               pQtTri->insert (aTri);
               mVTri.push_back(aTri);
	  }
      }
}

void cTriangulFormelleImplem::Finish()
{
     Pt2dr aPC = (mP0+mP1)/2.0;
     REAL dMin = 1e10;

     for (INT aK=0 ; aK<INT(mNodes.size()) ; aK++)
     {
        tTFISom * aS = mNodes[aK];
	Pt2dr aP = aS->attr()->PosPt();
	REAL aDist = euclid(aP,aPC);
	if ((mSomCentrale ==0) || (aDist<dMin))
	{
            dMin = aDist;
            mSomCentrale = aS;
	}
     }

     ELISE_ASSERT(mSomCentrale != 0,"cTriangulFormelleImplem::Finish");
     aPC = mSomCentrale->attr()->PosPt();
     dMin = 1e10;

     for (INT aK=0 ; aK<INT(mNodes.size()) ; aK++)
     {
        tTFISom * aS = mNodes[aK];
	Pt2dr aP = aS->attr()->PosPt();
	REAL aDist = ElAbs(aPC.y-aP.y) - ElAbs(aPC.x-aP.x);
	if ((mSomVecHorz ==0) || (aDist<dMin))
	{
            dMin = aDist;
            mSomVecHorz = aS;
	}
     }
     ELISE_ASSERT(mSomVecHorz != 0,"cTriangulFormelleImplem::Finish");

     if (pW)
        pW->draw_seg
        (
           TPRI(mSomCentrale->attr()->PosPt()),
           TPRI( mSomVecHorz->attr()->PosPt()),
	   pW->pdisc()(P8COL::white)
        );
}

cTFI_AttrSom * cTriangulFormelleImplem::SomCentral() 
{
    return mSomCentrale->attr();
}

cTFI_AttrSom * cTriangulFormelleImplem::VecHorz() 
{
    return mSomVecHorz->attr();
}


/************************************************************/
/*                                                          */
/*    cTriangulFormelleImplem                               */
/*                                                          */
/************************************************************/

cTriangulFormelle * cSetEqFormelles::NewTriangulFormelle
                      (
		         int aDim,
                         const std::list<Pt2dr> & aList,
                         REAL DMax,
                         ElDistortion22_Gen * aPosInit
                      )
{

    ELISE_ASSERT
    (
         aList.size() !=0 ,
	 "Empty list in cSetEqFormelles::NewTriangulFormelle"
    );

    Pt2dr aP0 =  aList.front();
    Pt2dr aP1 =  aList.front();
    for 
    (
        std::list<Pt2dr>::const_iterator it=aList.begin();
        it != aList.end() ; 
        it++
    )
    {
       aP0.SetInf(*it);
       aP1.SetSup(*it);
    }

    cTriangulFormelleImplem * aTri = new cTriangulFormelleImplem
	                                 (aDim,*this,Box2dr(aP0,aP1),DMax);

    std::vector<double> aV(aDim,0.0);
    for 
    (
        std::list<Pt2dr>::const_iterator it=aList.begin();
        it != aList.end() ; 
        it++
    )
    {
        if (aDim==2)
	{
           Pt2dr aQ = *it;
           if (aPosInit)
               aQ = aPosInit->Direct(aQ);
	   aV[0] = aQ.x;
	   aV[1] = aQ.y;
        }
	aTri->AddSom(*it,aV);
    }

    aTri->Triangulate();
    aTri->Finish();

    aTri->CloseEEF();
    AddObj2Kill(aTri);

    return aTri;
}


cTriangulFormelle * cSetEqFormelles::NewTriangulFormelle
                      (
		         int aDim,
                         Box2dr aBox,INT aNb,REAL DMax,
                         ElDistortion22_Gen * aPosInit
                      )
{
     std::list<Pt2dr>  aL;
     for (INT aKx=0 ; aKx<= aNb ; aKx++)
     {
         for (INT aKy=0 ; aKy<= aNb ; aKy++)
         {
             aL.push_back
             (
                  aBox.FromCoordLoc
		  (
		      Pt2dr(aKx/REAL(aNb),aKy/REAL(aNb))
		  )
             );
         }
     }
     return NewTriangulFormelle(aDim,aL,DMax,aPosInit);
}

cTriangulFormelle * cSetEqFormelles::NewTriangulFormelleUnitaire(int aDim)
{
    std::list<Pt2dr> aList;
    aList.push_back(Pt2dr(100,100));
    aList.push_back(Pt2dr(700,100));
    aList.push_back(Pt2dr(100,700));

    return NewTriangulFormelle(aDim,aList,2000);
}


void cTriangulFormelle::Test()
{
	#if (DEBUG_INTERNAL)
		All_Memo_counter MC_INIT;
		stow_memory_counter(MC_INIT);
	#endif

    {
       cSetEqFormelles aSet;

       /*
       std::list<Pt2dr> aL;
       for (INT aK=0 ; aK < 300 ; aK++)
           aL.push_back(Pt2dr(10,10)+Pt2dr(780*NRrandom3(),780*NRrandom3()));
       cTriangulFormelle * aTri = aSet.NewTriangulFormelle(aL);
       */
       cTriangulFormelle * aTri = aSet.NewTriangulFormelle
       (
           2,
           Box2dr(Pt2dr(100,100),Pt2dr(700,700)),
           10,
	   1.9/10.0
       );
       /*
       cTriangulFormelle * aTri = aSet.NewTriangulFormelleUnitaire();
       */

       while (true)
             aTri->TestOneTri();
    }
    // verif_memory_state(MC_INIT);
}


/*************************************************/
/*                                               */
/*                                               */
/*************************************************/

/*
class cInterfArgTrianguDensVar 
{
    public :
        cInterfArgTrianguDensVar
        (
              
        );
              Box2dr aBox,INT aNb,REAL DMax,
              ElDistortion22_Gen * aPosInit
        );
    private :
};

class cImplemArgTrianguDensVar 
{
    public :
    private :
};
*/






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
