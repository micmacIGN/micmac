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
#include <map>

// cTestMinimForm joue + ou - le role de namespace
class cTestMinimForm 
{
	public :
             typedef Fonc_Num            tFormule;
             typedef Pt2d<tFormule>      tPt2dF;
             typedef ElMatrix<tFormule>  tMatF;
};

class cSomMaillage;
class cTriangMaillage;

/*********************************************************************/
/*                                                                   */
/*                  cSomMaillage                                     */
/*                                                                   */
/*********************************************************************/

class cSomMaillage : public cTestMinimForm
{
    public :
	    // Pour creer un point "rigide"
	    
	    // Pour creer un point "mobile"
	    cSomMaillage(Pt2di aP0,Pt2dr aPInit,bool Rig,AllocateurDInconnues & anAloc) :
	         mP0    (aP0),
	         mPRes (aPInit),
		 mPInc  (   
                            Rig ?
                            tPt2dF(mPRes.x,mPRes.y):
                            anAloc.NewPt2(&mPRes.x,&mPRes.y)
			),
                 mRig   (Rig)
	    {
                static INT cpt =0;
                cpt++;
                if (cpt<=8)
                {
                    ncout() << "Point " << cpt << " : ";
                    ncout() << "X = ["; mPInc.x.show(cout) ; cout << "]";
                    ncout() << "Y = ["; mPInc.y.show(cout) ; cout << "]";
                    ncout() << "\n";
                }
	    }

	    const Pt2dr & P0()   const {return mP0;}
	    const tPt2dF & PInc() const {return mPInc;}
	    const Pt2dr & PRes()   const {return mPRes;}

            void Show(Video_Win aW);
    private :

	 Pt2dr            mP0;
	 Pt2dr            mPRes;
	 tPt2dF           mPInc;
         bool             mRig;
};

void cSomMaillage::Show(Video_Win aW)
{
   if (mRig)
      aW.draw_circle_abs(mPRes,3.0,aW.pdisc()(P8COL::red));
}

/*********************************************************************/
/*                                                                   */
/*                   TriangMaillage                                  */
/*                                                                   */
/*********************************************************************/

   // NOTATION :
   //
   //
   // soit P0,P1,P2 un triangle
   // soit U = P1-P0, V = P2-P0 (U,V) est la "base" du triangle
   //
   // soit (i,j) la base commune
   //
   //
  
class cTriangMaillage  : public cTestMinimForm
{
    public :
	    cTriangMaillage
	    (
	         cSomMaillage & aS0,
	         cSomMaillage & aS1,
	         cSomMaillage & aS2
	    ) :
	      mS0 (aS0),
	      mS1 (aS1),
	      mS2 (aS2)
	    {
                static INT cpt =0;
                cpt++;
                if (cpt<=2)
                {
                    cout << "Triang : " << cpt << " \n ";

                    tMatF aRes = MatFromCol(UInc(),VInc());
                    cout << "    MatUV(0,0) " ; aRes(0,0).show(cout); cout << "\n";

	            tMatF aMat =  MatDef() ;
                    cout << "    MatIJ(0,0) " ; aMat(0,0).show(cout); cout << "\n";

                    cout << "   E = ["; EId().show(cout) ; cout << "]";
                    cout << "\n";
                }
	    }

	    tMatF  MatDef() const;
	    tFormule EId() const; // Energie / identite
            REAL Surf() const
            {
                 return ElAbs(U0()^V0()); // ^ : produit vectoriel
            }


            void Show(Video_Win aW);

	    tFormule ERegul(const cTriangMaillage & T2) const; // Energie / identite

    private :

	    tPt2dF UInc() const {return mS1.PInc()-mS0.PInc();}
	    tPt2dF VInc() const {return mS2.PInc()-mS0.PInc();}

	    Pt2dr U0() const {return mS1.P0()-mS0.P0();}
	    Pt2dr V0() const {return mS2.P0()-mS0.P0();}

	    cSomMaillage & mS0;

	    cSomMaillage & mS1;
	    cSomMaillage & mS2;
};



void cTriangMaillage::Show(Video_Win aW)
{
    Col_Pal aCol (aW.pdisc()(P8COL::black));

    aW.draw_seg(mS0.PRes(),mS1.PRes(),aCol);
    aW.draw_seg(mS1.PRes(),mS2.PRes(),aCol);
    aW.draw_seg(mS2.PRes(),mS0.PRes(),aCol);
}

cTestMinimForm::tMatF cTriangMaillage::MatDef() const
{

   tMatF aRes = MatFromCol ( UInc(), VInc());
   //  Maintenant aRes contient l'expression
   //  formelle qui representent les images de U et V
   //  dans la base i,j
   //  Pour en faire une matrice de deformation il
   //  faut revenir aux images i,j dans la base i,j




   // aMChb = Matrice de changement de base
   ElMatrix<REAL> aMChb = MatFromCol
	               (
			   mS1.P0()-mS0.P0(),
			   mS2.P0()-mS0.P0()
		       );
   aMChb = gaussj(aMChb); // gaussj =renvoie l'inverse
   // Il suffit maintenant de faire aRes * aMChb


    return  aRes
            * ToMatForm(aMChb);

    // ToMatForm : transforme une matrice de valeurs reelles
    // en un matrice dont les elements sont des expressions
    // formelle, ces expressions formelles sont les 
    // fonctions contantes correspondant aux elements de la matrice
    // initiales
}

cTestMinimForm::tFormule cTriangMaillage::EId() const
{
    tMatF aMDef = MatDef();

    return ( Square(aMDef(0,0)-1)
           + Square(aMDef(1,1)-1)
           + Square(aMDef(0,1))
           + Square(aMDef(1,0))) * Surf();

}

cTestMinimForm::tFormule 
     cTriangMaillage:: ERegul(const cTriangMaillage & T2) const
{
    REAL s1 = Surf();
    REAL s2 = T2.Surf();

    tMatF aM1 = MatDef();
    tMatF aM2 = T2.MatDef();

    return    (s1+s2)
	    * (
	           Square(aM1(0,0)-aM2(0,0))
	         + Square(aM1(1,0)-aM2(1,0))
	         + Square(aM1(0,1)-aM2(0,1))
	         + Square(aM1(1,1)-aM2(1,1))
	      );
}

/*********************************************************************/
/*                                                                   */
/*                   cArcMaillage                                    */
/*                                                                   */
/*********************************************************************/

class cArcMaillageNor : public cTestMinimForm 
{
	public :
                cArcMaillageNor
                (
                    cSomMaillage *  s1,
                    cSomMaillage *  s2
                )  :
		   mS1 ((s1 >  s2) ? s1 : s2),
		   mS2 ((s1 <= s2) ? s1 : s2)
		{
		}
	        bool operator < (const cArcMaillageNor anA2) const
		{
		  if (mS1< anA2.mS1) return true;
		  if (mS1> anA2.mS1) return false;

		  if (mS2< anA2.mS2) return true;
		  if (mS2> anA2.mS2) return false;

		  return false;
		}
        private :
            cSomMaillage *  mS1;
            cSomMaillage *  mS2;

};

class cCpleTriVois  : public cTestMinimForm
{
    public :
	    cCpleTriVois () : mt1(0),mt2(0) {}

	    void NewTri ( cTriangMaillage * aTri)
	    {
		 if (mt1==0)
                     mt1 = aTri;
		 else if (mt2==0)
                     mt2 = aTri;
		 else 
                    ELISE_ASSERT(false,"Plus de 2 triangle/arc");

	    }

	    const cTriangMaillage & T1() const
	    {
		    AssertOk();
		    return *mt1;
	    }
	    const cTriangMaillage & T2() const
	    {
		    AssertOk();
		    return *mt2;
	    }
	    bool IsOk() const {return (mt1!=0)&&(mt2!=0);}

    private :			 
	    void AssertOk() const
	    {
                ELISE_ASSERT(IsOk(),"Incomplete Arc access");
	    }
	    cTriangMaillage * mt1;
	    cTriangMaillage * mt2;
};

/*********************************************************************/
/*                                                                   */
/*                   cMaillageSimple                                 */
/*                                                                   */
/*********************************************************************/

class cMaillageSimple : public cTestMinimForm
{
      public :
          ~cMaillageSimple();
          cMaillageSimple
          (
              INT aNb,
              INT aPerRigid,
	      bool QuadSpec
          );

          void CalculEnergie();
          void MinimizeEnergie();
          void Show( Video_Win);

      private :
          typedef std::list<cTriangMaillage *>  tSetTri;
          typedef std::list<cSomMaillage *>     tSetSom;
          typedef std::map<cArcMaillageNor,cCpleTriVois> tSetArc;


          void AddTriangle(cSomMaillage *,cSomMaillage *,cSomMaillage *);

          AllocateurDInconnues mAllocInc;
          INT mNb;
          std::list<cSomMaillage *>     mSoms;
          std::list<cTriangMaillage *>  mTrian;
	  tSetArc                       mArcs;

          cOptimSommeFormelle           * mOptimizer;
	  bool                          mQuadSpec;
};

void cMaillageSimple::AddTriangle(cSomMaillage *pS0,cSomMaillage *pS1,cSomMaillage *pS2)
{
   cTriangMaillage * pTri = new cTriangMaillage(*pS0,*pS1,*pS2);
   mTrian.push_back(pTri);
   mArcs[cArcMaillageNor(pS0,pS1)].NewTri(pTri);
   mArcs[cArcMaillageNor(pS1,pS2)].NewTri(pTri);
   mArcs[cArcMaillageNor(pS2,pS0)].NewTri(pTri);
}

template <class aType> void DeleteAllPointer( aType & aCont)
{
	
     for 
     (
          typename aType::iterator anIt = aCont.begin(); 
         anIt != aCont.end() ; 
         anIt++
     )
        delete *anIt;
}


cMaillageSimple::~cMaillageSimple()
{
    delete mOptimizer;
    DeleteAllPointer(mSoms);
    DeleteAllPointer(mTrian);
}

cMaillageSimple::cMaillageSimple(INT aNb,INT aPerRigid,bool QuadSpec) :
      mNb        (aNb),
      mOptimizer (0),
      mQuadSpec  (QuadSpec)
{
    std::vector<std::vector<cSomMaillage *> >  
        vSoms (aNb+1,std::vector<cSomMaillage *>(aNb+1,(cSomMaillage *)0));// [1]

    for (INT y=0 ; y<= aNb ; y++)
    {
       for (INT x=0 ; x<= aNb ; x++)
       {
           Pt2di aP0(x,y);
           if ((x%aPerRigid==0) && (y%aPerRigid==0))
           {
               Pt2dr aDec = Pt2dr(NRrandC(),NRrandC()) * (aPerRigid/4.0);
               vSoms[y][x] = new cSomMaillage(aP0,aP0+aDec,true,mAllocInc);
           }
           else
           {
               vSoms[y][x] = new cSomMaillage(aP0,aP0,false,mAllocInc);
           }
       }
    }

	{
    for (INT y=0 ; y<= aNb ; y++)
       for (INT x=0 ; x<= aNb ; x++)
           mSoms.push_back(vSoms[y][x]);    
	}

	{
    for (INT y=0 ; y< aNb ; y++)
       for (INT x=0 ; x< aNb ; x++)
       {
           AddTriangle(vSoms[y][x],vSoms[y+1][x],vSoms[y][x+1]);
           AddTriangle(vSoms[y+1][x+1],vSoms[y+1][x],vSoms[y][x+1]);
       }
	}
}

void cMaillageSimple::CalculEnergie()
{
     delete mOptimizer;
     mOptimizer = new cOptimSommeFormelle(mAllocInc.CurInc());
     for 
     (
         tSetTri::const_iterator anIt = mTrian.begin(); 
         anIt != mTrian.end() ; 
         anIt++
     )
     {
           mOptimizer->Add((*anIt)->EId(),mQuadSpec);
     }

     INT aCptArc = 0;
	 {
     for 
     (
         tSetArc::const_iterator anIt = mArcs.begin(); 
         anIt != mArcs.end() ; 
         anIt++
     )
     {
	     if (anIt->second.IsOk())
	     {
		const cTriangMaillage & aT1 = anIt->second.T1();
		const cTriangMaillage & aT2 = anIt->second.T2();
		aCptArc++;

		mOptimizer->Add(1000*aT1.ERegul(aT2),mQuadSpec);
	     }
     }
	 }
     cout << "AnbArc = " << aCptArc << "\n";
}

void cMaillageSimple::Show(Video_Win aW)
{
     for 
     (
         tSetTri::const_iterator anIt = mTrian.begin(); 
         anIt != mTrian.end() ; 
         anIt++
     )
     {
           (*anIt)->Show(aW);
     }

	{
     for 
     (
         tSetSom::const_iterator anIt = mSoms.begin(); 
         anIt != mSoms.end() ; 
         anIt++
     )
     {
       (*anIt)->Show(aW);
     }
	}
}

void cMaillageSimple::MinimizeEnergie()
{
   PtsKD aP = mAllocInc.PInits();

  INT aNbIter = mOptimizer->GradConjMin(aP,1e-6,200);
  cout << "Nb Iter = " << aNbIter << "\n";

  mAllocInc.SetVars(aP.AdrX0());
}



/**************************************************/
/*                                                */
/*    FONCTIONS GLOBALES                          */
/*                                                */
/**************************************************/




void testMaille(INT aNbMaille,Video_Win  aW,bool QuadSpec)
{


     ElTimer aTimer;
     cMaillageSimple aMaillage(aNbMaille,5,QuadSpec);
     cout << "Sz : " << aNbMaille << " " << aTimer.uval() << "\n";

     aTimer.reinit();
     aMaillage.CalculEnergie();
     cout << "Calc E  : " <<  aTimer.uval() << "\n";



     aTimer.reinit();
     aMaillage.MinimizeEnergie();
     cout << "Minim E  : " <<  aTimer.uval() << "\n";


     ELISE_COPY(aW.all_pts(),P8COL::white,aW.odisc());

     REAL aZ = aW.sz().x / (aNbMaille + 3.0);
     aW = aW.chc(Pt2dr(-1,-1),Pt2dr(aZ,aZ));


     aMaillage.Show(aW);

     cout << "\n";
     getchar();
}

void testElementsFinis()
{
        Video_Win  aW = Video_Win::WStd(Pt2di(900,900),1.0);



	
	testMaille(10,aW,true);
/*
	testMaille(10,aW);
	testMaille(20,aW);
	testMaille(30,aW);
	testMaille(40,aW);
	testMaille(50,aW);

	testMaille(100,aW);
*/
}

/*
    [1]  
      vector<Type>(int Nb,const Type & T) => construit un vecteur avec
      Nb element copie de T

      donc std::vector<cSomMaillage *>(aNb+1,(cSomMaillage *)0)
*/









