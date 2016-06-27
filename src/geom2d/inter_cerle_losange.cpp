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



/*
    Calcul de l'intersection entre un cercle et un losange,
    une ellispe et un carre.
 
    ICL : Intersection Cerle Losange
*/


class cICL_Arc
{
	public  :
		cICL_Arc 
                (
                     Pt2dr aP0,
                     bool  isP0OnCerle,
                     Pt2dr aP1,
                     bool  isP1OnCerle
                );

		const Pt2dr & P0 ()          const;
		const bool  & IsP0OnCercle() const;
		const Pt2dr & P1 ()          const;
		const bool  & IsP1OnCercle() const;
	private  :
		Pt2dr mP0;
		bool  mIsP0OnCercle;
		Pt2dr mP1;
		bool  mIsP1OnCercle;
};




class cICL_Courbe
{
	public :

	   void AddSeg
		(
		      Pt2dr P0,
		      bool  isP0OnCerle,
		      Pt2dr P1,
		      bool  isP1OnCerle
		);
           void DoAll( const std::vector<Pt2dr> &  VPtsInit);

           cICL_Courbe(bool Visu);
           std::vector<Pt2dr> GetPoly(INT NbPtsMax,bool ForceLosange);
	   bool WithW() const;

	   void SetVisu();
	   void SetNoVisu();
	   REAL Surf() const;
	   REAL SurfDigit(REAL Scale) const; // Pour Verif
	private :
	   Pt2dr ToLoc(Pt2dr aP) const;

	   REAL                  mScale;
	   Pt2dr                 mP0;
	   std::vector<Pt2dr>    mVPts;
	   std::vector<cICL_Arc> mVArcs;
           Video_Win             * pW;
	   Video_Win             * pWSauv;
	   REAL                  mSurf;
           bool                  mDerivNulles;
};

// Renvoie l'intersection entre le cerle unite  et un segment de droite
// renvoie toujours un couple de point, si Res>0 deux vrais points,
// si Res==0 un point double, si Res <0 un point double correspondant
// a la projection du cercle sur le segment
// Le premier point est le plus pres de Seg.p0()
REAL IntersecSegCercle(const SegComp &aSeg,Pt2dr & Q0,Pt2dr & Q1);

/*************************************************/
/*                                               */
/*             ::                                */
/*                                               */
/*************************************************/

REAL IntersecSegCercle(const SegComp &aSeg,Pt2dr & Q0,Pt2dr & Q1)
{
    Pt2dr T = aSeg.tangente();
    Pt2dr p0 = aSeg.p0();

    REAL a =  square_euclid(T);
    REAL b = 2 * scal(T,p0);
    REAL c = square_euclid(p0) - 1;
    REAL delta = ElSquare(b) - 4 * a * c;
    REAL SqDelta = sqrt(ElMax(0.0,delta));

    Q0  = p0 + T*((-b-SqDelta)/(2*a));
    Q1  = p0 + T*((-b+SqDelta)/(2*a));

    return delta;
}

/*************************************************/
/*                                               */
/*              cICL_Arc                         */
/*                                               */
/*************************************************/
 
cICL_Arc::cICL_Arc 
(
      Pt2dr aP0,
      bool  isP0OnCerle,
      Pt2dr aP1,
      bool  isP1OnCerle
)  :
   mP0 (aP0),
   mIsP0OnCercle (isP0OnCerle),
   mP1 (aP1),
   mIsP1OnCercle (isP1OnCerle)
{
}
const Pt2dr & cICL_Arc::P0 ()           const {return mP0;}
const bool  & cICL_Arc::IsP0OnCercle()  const {return mIsP0OnCercle;}
const Pt2dr & cICL_Arc::P1 ()           const {return mP1;}
const bool  & cICL_Arc::IsP1OnCercle()  const {return mIsP1OnCercle;}

/*************************************************/
/*                                               */
/*             cICL_Courbe                       */
/*                                               */
/*************************************************/
 
cICL_Courbe::cICL_Courbe(bool Visu) :
   mScale (100.0),
   mP0    (mScale * 3,mScale * 3),
   pW     (0),
   pWSauv (0)
{
    if (Visu)
       SetVisu();
}

void cICL_Courbe::SetVisu()
{
     if (pWSauv == 0)
        pWSauv = Video_Win::PtrWStd(Pt2di(mP0*2),1.0);
     pW = pWSauv;
}

void cICL_Courbe::SetNoVisu()
{
	pW = 0;
}


REAL cICL_Courbe::Surf() const
{
    return mSurf;
}
	   

Pt2dr cICL_Courbe::ToLoc(Pt2dr aP) const
{
	return mP0 + aP*mScale;
}

void cICL_Courbe::AddSeg
                  (
		      Pt2dr P0,
		      bool  isP0OnCerle,
		      Pt2dr P1,
		      bool  isP1OnCerle
                  )
{		  
    if (pW)
    {
        pW->draw_arrow
        (
	    ToLoc(P0),
            ToLoc(P1),
            Line_St(pW->pdisc()(P8COL::blue),2),
            Line_St(pW->pdisc()(P8COL::blue),2),
	    10.0
        );
        pW->draw_circle_loc
        (
           ToLoc(P0),
           5.0,
           pW->pdisc()(isP0OnCerle?P8COL::yellow : P8COL::green)
        );
        pW->draw_circle_loc
        (
           ToLoc(P1),
           5.0,
           pW->pdisc()(isP1OnCerle?P8COL::yellow : P8COL::green)
        );
    }

    mVArcs.push_back(cICL_Arc(P0,isP0OnCerle,P1,isP1OnCerle));
}

bool cICL_Courbe::WithW() const
{
     return pW != 0;
}


std::vector<Pt2dr> cICL_Courbe::GetPoly(INT NbPtsMax,bool ForceLosange)
{
    ELISE_ASSERT(pW!=0,"cICL_Courbe::GetPoly()");
    static bool First = true;
    if (! First)
        pW->clik_in();
    First = false;
    pW->clear();
    pW->draw_circle_loc
    (
        ToLoc(Pt2dr(0,0)),
        mScale,
        pW->pdisc()(P8COL::red)
    );
    std::vector<Pt2dr> aRes;

    while (true)
    {
        Clik aCl = pW->clik_in();
	Pt2dr aP = (Pt2dr(aCl._pt)-mP0) / mScale;

	if (NbPtsMax-1 == INT(aRes.size()))
            aCl._b = 3;

	if (ForceLosange && (aRes.size()==3))
        {
            aP = aRes[0]+ aRes[2] - aRes[1];
            aCl._b = 3;
	}
        pW->draw_circle_loc(ToLoc(aP),3.0,pW->pdisc()(P8COL::green));
        
	 if (! aRes.empty())
            pW->draw_seg(ToLoc(aRes.back()),ToLoc(aP),pW->pdisc()(P8COL::green));

	 aRes.push_back(aP);

	 if (aCl._b == 3)
	 {
            pW->draw_seg(ToLoc(aRes[0]),ToLoc(aP),pW->pdisc()(P8COL::green));
	    return aRes;
	 }
    }
    return aRes;
}


void cICL_Courbe::DoAll(const std::vector<Pt2dr> &  VPtsInit)
{

   mVPts = VPtsInit;
   
   if (surf_or_poly(mVPts) < 0)
      std::reverse(mVPts.begin(),mVPts.end());

   INT aNbPts = (INT) mVPts.size();
   mVArcs.clear();
   for (INT aK=0 ; aK < aNbPts ; aK++)
   {
       Pt2dr aP1 = mVPts[aK];
       Pt2dr aP2 = mVPts[(aK+1)%aNbPts];

       bool isP1Inside = (square_euclid(aP1) < 1.0);
       bool isP2Inside = (square_euclid(aP2) < 1.0);

       if (isP1Inside && isP2Inside)
       {
          AddSeg(aP1,false,aP2,false);
       }
       else
       {
          Pt2dr aQ1,aQ2;
	  SegComp aS12(aP1,aP2);
	  REAL D = IntersecSegCercle(aS12,aQ1,aQ2);

          if ((! isP1Inside) && (! isP2Inside))
	  {
             if (D>0)
	     {
		 Pt2dr Mil = (aQ1+aQ2)/2.0;
		 if (aS12.in_bande(Mil,SegComp::seg))
                    AddSeg(aQ1,true,aQ2,true);
	     }
	  }
          else if (isP1Inside && (! isP2Inside))
               AddSeg(aP1,false,aQ2,true);
	  else
              AddSeg(aQ1,true,aP2,false);
       }
   }
			       

   mSurf = 0;

   if (mVArcs.empty())
   {
      mDerivNulles = true;
      if (PointInPoly(mVPts,Pt2dr(0,0)))
         mSurf = PI;
      else
         mSurf = 0.0;
   }
   else
   {
       mDerivNulles = false;
       INT aNbA = (INT) mVArcs.size() ;
       for (INT aK=0 ; aK< aNbA ; aK++)
       {
            cICL_Arc & anA = mVArcs[aK];
            mSurf +=  (anA.P0() ^ anA.P1()) / 2.0;
            if (anA.IsP1OnCercle())
            {
                cICL_Arc & nextA = mVArcs[(aK+1)%aNbA];
		if (nextA.IsP0OnCercle())
		{
                       REAL Ang = angle(anA.P1(),nextA.P0());
		       if (Ang <0)
                          Ang += 2 * PI;
		       mSurf += Ang/2.0;
		}
            }
       }
   }

   /*

   if (true)
   {
       INT aNbPts = mVPts.size();
       ElList<Pt2di> lPts;
       REAL scale = 1000.0;
       for (int aK=0 ; aK<aNbPts ; aK++)
          lPts = lPts + Pt2di(mVPts[aK]*scale);

       INT Ok;
       ELISE_COPY
       (
	   polygone(lPts),
	   (FX*FX+FY*FY) < scale*scale,
	   sigma(Ok)
       );
       cout << " SURF = " << mSurf 
            << " SDig = " << Ok/ElSquare(scale) 
            << "\n";
   }
   */
		   
}

REAL cICL_Courbe::SurfDigit(REAL Scale) const // Pour Verif
{
    INT aNbPts = (INT) mVPts.size();
    ElList<Pt2di> lPts;
    for (int aK=0 ; aK<aNbPts ; aK++)
          lPts = lPts + Pt2di(mVPts[aK]*Scale);

    INT Ok;
    ELISE_COPY
    (
         polygone(lPts),
         (FX*FX+FY*FY) < Scale*Scale,
         sigma(Ok)
    );

    return Ok/ElSquare(Scale);
}

/*************************************************/
/*                                               */
/*             cArcICL                           */
/*                                               */
/*************************************************/


REAL IntersectionCercleUniteLosange
     (
          Pt2dr aP0,
          Pt2dr aP1,
          Pt2dr aP2
     )
{
   static cICL_Courbe aCurvAct(false);
   static std::vector<Pt2dr> VPts;
   VPts.clear();
   VPts.push_back(aP0);
   VPts.push_back(aP1);
   VPts.push_back(aP1+aP2-aP0);
   VPts.push_back(aP2);

   aCurvAct.DoAll(VPts);
   /*
   cout << " SURF   = " << aCurvAct.Surf()
	<< " S(100) = " << aCurvAct.SurfDigit(100.0)
	<< " S(1000) = " << aCurvAct.SurfDigit(1000.0)
	<< "\n";
   */
   return aCurvAct.Surf();
}


// ELLIPSE DEFINIE PAR 
//   Son Centre CentreEllipse,
//   L'equation (AX+BY)2 + (BX+CY)2 = 1
//   Image reciproque du cercle unite par 
//
//           A B
//     M =   B C
//

Pt2dr ImAppSym(REAL A,REAL B,REAL C,Pt2dr aP)
{
	return Pt2dr(aP.x*A+aP.y*B,aP.x*B+aP.y*C);
}

REAL SurfIER
     (
          Pt2dr CentreEllipse,
	  REAL  A,
	  REAL  B,
	  REAL  C,
	  Pt2dr Corner0,
	  Pt2dr Corner1
     )
{
    Corner0 -= CentreEllipse;
    Corner1 -= CentreEllipse;
   
    Pt2dr P0 = ImAppSym(A,B,C,Corner0);
    Pt2dr P1 = ImAppSym(A,B,C,Pt2dr(Corner1.x,Corner0.y));
    Pt2dr P2 = ImAppSym(A,B,C,Pt2dr(Corner0.x,Corner1.y));

    return IntersectionCercleUniteLosange(P0,P1,P2) / (A*C-B*B);
}


#define EpsABC 1e-5

REAL DerASurfIER(Pt2dr CEl,REAL A,REAL B,REAL C,Pt2dr P0,Pt2dr P1)
{
   REAL Eps = EpsABC * (ElAbs(A)+ElAbs(B)+ElAbs(C));
   return (
             SurfIER(CEl,A+Eps,B,C,P0,P1)
            -SurfIER(CEl,A-Eps,B,C,P0,P1)
           )  / (2*Eps);
}

REAL DerBSurfIER(Pt2dr CEl,REAL A,REAL B,REAL C,Pt2dr P0,Pt2dr P1)
{
   REAL Eps = EpsABC * (ElAbs(A)+ElAbs(B)+ElAbs(C));
   return (
             SurfIER(CEl,A,B+Eps,C,P0,P1)
            -SurfIER(CEl,A,B-Eps,C,P0,P1)
           )  / (2*Eps);
}

REAL DerCSurfIER(Pt2dr CEl,REAL A,REAL B,REAL C,Pt2dr P0,Pt2dr P1)
{
   REAL Eps = EpsABC * (ElAbs(A)+ElAbs(B)+ElAbs(C));
   return (
             SurfIER(CEl,A,B,C+Eps,P0,P1)
            -SurfIER(CEl,A,B,C-Eps,P0,P1)
           )  / (2*Eps);
}

#define EpsXY 1e-5

REAL DerCElXSurfIER(Pt2dr CEl,REAL A,REAL B,REAL C,Pt2dr P0,Pt2dr P1)
{
   return (
             SurfIER(Pt2dr(CEl.x+EpsXY,CEl.y),A,B,C,P0,P1)
            -SurfIER(Pt2dr(CEl.x-EpsXY,CEl.y),A,B,C,P0,P1)
           )  / (2*EpsXY);
}
REAL DerCElYSurfIER(Pt2dr CEl,REAL A,REAL B,REAL C,Pt2dr P0,Pt2dr P1)
{
   return (
             SurfIER(Pt2dr(CEl.x,CEl.y+EpsXY),A,B,C,P0,P1)
            -SurfIER(Pt2dr(CEl.x,CEl.y-EpsXY),A,B,C,P0,P1)
           )  / (2*EpsXY);
}

REAL DerP0XSurfIER(Pt2dr CEl,REAL A,REAL B,REAL C,Pt2dr P0,Pt2dr P1)
{
   return (
              SurfIER(CEl,A,B,C,Pt2dr(P0.x+EpsXY,P0.y),P1)
            - SurfIER(CEl,A,B,C,Pt2dr(P0.x-EpsXY,P0.y),P1)
           )  / (2*EpsXY);
}

REAL DerP0YSurfIER(Pt2dr CEl,REAL A,REAL B,REAL C,Pt2dr P0,Pt2dr P1)
{
   return (
              SurfIER(CEl,A,B,C,Pt2dr(P0.x,P0.y+EpsXY),P1)
            - SurfIER(CEl,A,B,C,Pt2dr(P0.x,P0.y-EpsXY),P1)
           )  / (2*EpsXY);
}

REAL DerP1XSurfIER(Pt2dr CEl,REAL A,REAL B,REAL C,Pt2dr P0,Pt2dr P1)
{
   return (
              SurfIER(CEl,A,B,C,P0,Pt2dr(P1.x+EpsXY,P1.y))
            - SurfIER(CEl,A,B,C,P0,Pt2dr(P1.x-EpsXY,P1.y))
           )  / (2*EpsXY);
}
REAL DerP1YSurfIER(Pt2dr CEl,REAL A,REAL B,REAL C,Pt2dr P0,Pt2dr P1)
{
   return (
              SurfIER(CEl,A,B,C,P0,Pt2dr(P1.x,P1.y+EpsXY))
            - SurfIER(CEl,A,B,C,P0,Pt2dr(P1.x,P1.y-EpsXY))
           )  / (2*EpsXY);
}

typedef REAL (* TyFctrSurfIER)(Pt2dr CEl,REAL A,REAL B,REAL C,Pt2dr P0,Pt2dr P1);

class SurfIER_Fonc_Num_Not_Comp : public Fonc_Num_Not_Comp
{
    public :
	    SurfIER_Fonc_Num_Not_Comp
            (
	        std::string &aName,
		TyFctrSurfIER aFctr,
	        Pt2d<Fonc_Num> aCel,
		Fonc_Num aA,Fonc_Num aB,Fonc_Num aC,
		Pt2d<Fonc_Num> aP0,Pt2d<Fonc_Num> aP1
            );
    private :
       Pt2d<Fonc_Num>  fCEl;
       Fonc_Num        fA;
       Fonc_Num        fB;
       Fonc_Num        fC;
       Pt2d<Fonc_Num>  fP0;
       Pt2d<Fonc_Num>  fP1;
       std::string     mName;
       TyFctrSurfIER   mFctr;

       Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp &)
       {
               ELISE_ASSERT(false,"No  SurfIER_Fonc_Num_Not_Comp::compute");
               return 0;
       }
       bool integral_fonc(bool integral_flux) const {return false;}
       INT dimf_out() const {return 1;}
       virtual void VarDerNN(ElGrowingSetInd &) const 
       {
           ELISE_ASSERT(false,"No VarDerNN");
       }

       Fonc_Num FNF(const std::string &,TyFctrSurfIER) const;


       virtual REAL  ValFonc(const  PtsKD &  pts) const ;
       virtual Fonc_Num deriv(INT k) const ;
       virtual void compile (cElCompileFN &);

       Fonc_Num::tKindOfExpr KindOfExpr() 
       {
           return Fonc_Num::eIsSurfIER;
       }
       virtual INT CmpFormelIfSameKind(Fonc_Num_Not_Comp *);
};

INT SurfIER_Fonc_Num_Not_Comp::CmpFormelIfSameKind(Fonc_Num_Not_Comp * aF2)
{
   SurfIER_Fonc_Num_Not_Comp * pSFNC  = (SurfIER_Fonc_Num_Not_Comp *) aF2;

   INT res = CmpTertiare(mName,pSFNC->mName);

   if (res) return res;

   res = fCEl.x.CmpFormel(pSFNC->fCEl.x);
   if (res) return res;

   res = fCEl.y.CmpFormel(pSFNC->fCEl.y);
   if (res) return res;

   res = fA.CmpFormel(pSFNC->fA);
   if (res) return res;

   res = fB.CmpFormel(pSFNC->fB);
   if (res) return res;

   res = fC.CmpFormel(pSFNC->fC);
   if (res) return res;

   res = fP0.x.CmpFormel(pSFNC->fP0.x);
   if (res) return res;

   res = fP0.y.CmpFormel(pSFNC->fP0.y);
   if (res) return res;

   res = fP1.x.CmpFormel(pSFNC->fP1.x);
   if (res) return res;

   res = fP1.y.CmpFormel(pSFNC->fP1.y);
   if (res) return res;

   return 0;
}


SurfIER_Fonc_Num_Not_Comp::SurfIER_Fonc_Num_Not_Comp
(
      std::string &aName,
      TyFctrSurfIER aFctr,
      Pt2d<Fonc_Num> aCel,
      Fonc_Num aA,Fonc_Num aB,Fonc_Num aC,
      Pt2d<Fonc_Num> aP0,Pt2d<Fonc_Num> aP1
)  :
    fCEl   (aCel),
    fA     (aA),
    fB     (aB),
    fC     (aC),
    fP0    (aP0),
    fP1    (aP1),
    mName  (aName),
    mFctr  (aFctr)
{
}


Fonc_Num    FN_SurfIER
            (
	        std::string aName,
		TyFctrSurfIER aFctr,
	        Pt2d<Fonc_Num> aCel,
		Fonc_Num aA,Fonc_Num aB,Fonc_Num aC,
		Pt2d<Fonc_Num> aP0,Pt2d<Fonc_Num> aP1
            )
{
	return new SurfIER_Fonc_Num_Not_Comp
		(
		     aName,aFctr,
		     aCel,aA,aB,aC,
		     aP0,aP1
		);
}
Fonc_Num    FN_SurfIER
            (
	        Pt2d<Fonc_Num> aCel,
		Fonc_Num aA,Fonc_Num aB,Fonc_Num aC,
		Pt2d<Fonc_Num> aP0,Pt2d<Fonc_Num> aP1
            )
{
     return FN_SurfIER("SurfIER",SurfIER,aCel,aA,aB,aC,aP0,aP1);
}

Fonc_Num SurfIER_Fonc_Num_Not_Comp::FNF
         (const std::string &aName,TyFctrSurfIER aFctr) const
{
	return FN_SurfIER(aName,aFctr,fCEl,fA,fB,fC,fP0,fP1);
}


Fonc_Num SurfIER_Fonc_Num_Not_Comp::deriv(INT k) const
{
    ELISE_ASSERT(mFctr==SurfIER,"SurfIER_Fonc_Num_Not_Comp::deriv");
    return 
	      fCEl.x.deriv(k) * FNF("DerCElXSurfIER",DerCElXSurfIER)
	    + fCEl.y.deriv(k) * FNF("DerCElYSurfIER",DerCElYSurfIER)
	    + fA.deriv(k)     * FNF("DerASurfIER",DerASurfIER)
	    + fB.deriv(k)     * FNF("DerBSurfIER",DerBSurfIER)
	    + fC.deriv(k)     * FNF("DerCSurfIER",DerCSurfIER)
	    + fP0.x.deriv(k)  * FNF("DerP0XSurfIER",DerP0XSurfIER)
	    + fP0.y.deriv(k)  * FNF("DerP0YSurfIER",DerP0YSurfIER)
	    + fP1.x.deriv(k)  * FNF("DerP1XSurfIER",DerP1XSurfIER)
	    + fP1.y.deriv(k)  * FNF("DerP1YSurfIER",DerP1YSurfIER);
}


REAL  SurfIER_Fonc_Num_Not_Comp::ValFonc(const  PtsKD &  pts) const
{
   return mFctr
	   (
	       Pt2dr(fCEl.x.ValFonc(pts),fCEl.y.ValFonc(pts)),
	       fA.ValFonc(pts),fB.ValFonc(pts),fC.ValFonc(pts),
	       Pt2dr(fP0.x.ValFonc(pts),fP0.y.ValFonc(pts)),
	       Pt2dr(fP1.x.ValFonc(pts),fP1.y.ValFonc(pts))
	   );
}

void SurfIER_Fonc_Num_Not_Comp::compile (cElCompileFN & anEnv)
{
    anEnv << mName << "("
	  << "Pt2dr(" << fCEl.x << "," << fCEl.y << ") " << ","
	  << fA << "," << fB  << "," << fC << ","
	  << "Pt2dr(" << fP0.x << "," << fP0.y << ") " << ","
	  << "Pt2dr(" << fP1.x << "," << fP1.y << ") " 
	  << ")" ;
}







// De l'image par une transfo affine d'un repere orthonorme
// au parametre ABC de l'ellipse passant par ce repere

void ImRON2ParmEllipse
     (
         REAL & A,
         REAL & B,
         REAL & C,
	 const Pt2dr & aV0,
	 const Pt2dr & aV1
     )
{
   ElMatrix<REAL> aMat(2,2);
   SetCol(aMat,0,aV0);
   SetCol(aMat,1,aV1);

   aMat = gaussj(aMat);
   aMat = aMat.transpose() * aMat;

   ElMatrix<REAL> aVecP(2,2);
   ElMatrix<REAL> aValP(2,2);

   jacobi_diag(aMat,aValP,aVecP);


    aValP(0,0) = sqrt(aValP(0,0));
    aValP(1,1) = sqrt(aValP(1,1));

    ElMatrix<REAL> aABC = aVecP * aValP * aVecP.transpose();

    A = aABC(0,0);
    B = aABC(1,0);
    C = aABC(1,1);
}

// Des parametres A,B,C de l'equation aux
// parametres "physiques"


bool EllipseEq2ParamPhys
     (
         REAL  & V1,
         REAL  & V2,
         REAL  & teta,
         REAL  A,
         REAL  B,
         REAL  C
     )
{
   ElMatrix<REAL> aMat(2,2);
   aMat(0,0) = A;
   aMat(1,0) = B;
   aMat(0,1) = B;
   aMat(1,1) = C;

   ElMatrix<REAL> aVecP(2,2);
   ElMatrix<REAL> aValP(2,2);

   jacobi_diag(aMat,aValP,aVecP);
   if ((aValP(0,0) <=0) || (aValP(1,1) <= 0))
       return false;
   V1 = 1 / aValP(0,0);
   V2 = 1 / aValP(1,1);
   teta = angle(Pt2dr(aVecP(0,0),aVecP(0,1)));

   if (V1 < V2)
   {
      ElSwap(V1,V2);
      teta += PI/2;
      if (teta> PI)
         teta -= PI;
   }

   return true;
}

void InvertParamEllipse
     (
        REAL & A,  REAL & B,  REAL & C ,
        REAL  A0,  REAL  B0,  REAL   C0
     )
{
   REAL Delta = A0*C0-B0*B0;
   B = -B0/Delta;
   A = C0 / Delta;
   C = A0 / Delta;
}

REAL  SimilariteEllipse(REAL A1,REAL B1,REAL C1,REAL A2,REAL B2,REAL C2)
{
	return sqrt
	       (	
		    (ElSquare(A1-A2) +2 *ElSquare(B1-B2)+ElSquare(C1-C2))
		 /  (
		            ElSquare(A1)/2 +ElSquare(B1)+ElSquare(C1)/2
		          + ElSquare(A2)/2 +ElSquare(B2)+ElSquare(C2)/2
		     ) 
	       );
}


Box2dr BoxEllipse(Pt2dr aCenter,REAL A,REAL B,REAL C)
{
     InvertParamEllipse(A,B,C,A,B,C);
     REAL lX = sqrt(A*A+B*B);
     REAL lY = sqrt(B*B +C*C);

     return Box2dr(aCenter-Pt2dr(lX,lY),aCenter+Pt2dr(lX,lY));
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
