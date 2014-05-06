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


/*******************************************/
/*                                         */
/*       MajEncadrRoots                    */
/*                                         */
/*******************************************/

/*
    Etant donne un polynome reel aPol renvoit un reel X tel
    que [-X X] encadre (assez largement) l'intervalle ou peut se
    trouver les racines reelles

                 n
    Pol(x) = An x  * (1 + A(n-1)  +   A(n-2)  + ...
                         -------     ------2
                          An X        An  X

       On s'arrange pour que chaque terme (en dehors du 1) soit < a 1/2n
     en valeur absolue
*/

// Racine d'un reel supose >= 0
REAL IRoots(REAL val,INT exp)
{
    switch (exp)
    {
         case 0  : return 1.0;
         case 1  : return val;
         case 2  : return sqrt(val);
//         case 3  : return cbrt(val);
         default : return pow(val,1.0/exp);
    }
}

template <class Type> Type MajEncadrRoots(const ElPolynome<Type>  & aPol)
{
    Type res = 0;
    INT n = aPol.degre();
    Type An = aPol[n];
    ELISE_ASSERT(An!=0,"Coeff =0 in MajEncadrRoots");
    Type aCoeff = (2.0*n)/An;

    for (INT k=1 ; k<=n; k++)
        ElSetMax(res,IRoots(ElAbs(aPol[n-k]*aCoeff),k));

    return res;
}


template <class Type> class DichotSolvePolynone : public NROptF1vDer
{
    public :
        DichotSolvePolynone
        (
              const ElPolynome<Type>  & aPol,
              const ElPolynome<Type>  & aPolDeriv
        ) :
           mPol      (aPol),
           mPolDeriv (aPolDeriv)
        {
        }

        Type BracketSolve(Type a,Type b,bool & Ok,Type tol,INT ItMax);

    private :
       const ElPolynome<Type>  & mPol;
       const ElPolynome<Type>  & mPolDeriv;

       REAL NRF1v(REAL c)    {return mPol(c);}
       REAL DerNRF1v(REAL c) {return mPolDeriv(c);}
};


template <class Type> Type DichotSolvePolynone<Type>::BracketSolve
     (
        Type a,
        Type b,
        bool & Ok,
        Type tol,
        INT ItMax
     )
{
  Type ca = mPol(a);
  Type cb = mPol(b);
  Ok =   ((ca<0) && (cb>0))    ||  ((ca>0) && (cb<0));

  if (!Ok)
      return HUGE_VAL;

  Type res= rtsafe(a,b,tol,ItMax);

  ELISE_ASSERT((res>=a) && (res<=b), "Bad Bracketting in DichotSolvePolynone::BracketSolve ");
  return res;
}



/*
    Cette fonction s'appelle IntervVarOfPrimtive
    (et non RootsOf...) car afin d'etre homogen
    elle rajoute systematiquement les bornes sup
    en tete de vecteur resultats
*/

template<class Type>
void IntervVarOfPrimtive
     (
         ElSTDNS vector<Type> &  Roots,
         Type  BorneInf,
         Type  BorneSup,
         const ElPolynome<Type>  &aPol,
         Type                    tol,
         INT                     ItMax,
         bool                    push_borne
     )
{
     if (push_borne)
         Roots.push_back(BorneInf);
     if (aPol.degre() == 1)
     {
        Type a1 = aPol[1];
        if (a1 == 0) return;
        Type TheRoot = - aPol[0] / a1;
        if  ((TheRoot>BorneInf) && (TheRoot<BorneSup))
             Roots.push_back(TheRoot);
     }
     else
     {

         ElSTDNS vector<Type>  RootsDeriv;
         ElPolynome<Type> PolDeriv = aPol.deriv();
      

         IntervVarOfPrimtive
         (
             RootsDeriv,
             BorneInf,
             BorneSup,
             PolDeriv,
             tol,
             ItMax,
             true
         );
         for (INT k=1 ; k<(INT)RootsDeriv.size() ; k++)
         {
            DichotSolvePolynone<Type> aSolver(aPol,PolDeriv);
            bool Ok;
            Type aRoot = aSolver.BracketSolve(RootsDeriv[k-1],RootsDeriv[k],Ok,tol,ItMax);
            if (Ok)
               Roots.push_back(aRoot); 
         }
     }
     if (push_borne)
        Roots.push_back(BorneSup);
}



bool BugPoly = false;


template <class Type> void  
     RealRootsOfRealPolynome
     (
         ElSTDNS vector<Type> &  Sols,
         const ElPolynome<Type>  &aPol,
         Type                    tol,
         INT                     ItMax
     )
{
    Sols.clear();
    if (aPol.degre() == 0) return;


    Type  Borne =  MajEncadrRoots(aPol);
    IntervVarOfPrimtive(Sols,-Borne,+Borne,aPol,tol,ItMax,false);
}


template  void  RealRootsOfRealPolynome<REAL>
                  (
                      ElSTDNS vector<REAL> &  Sols,
                      const ElPolynome<REAL>  &aPol,
                      REAL                    tol,
                      INT                     ItMax
                  );
template  void  RealRootsOfRealPolynome<float>
                  (
                      ElSTDNS vector<float> &  Sols,
                      const ElPolynome<float>  &aPol,
                      float                    tol,
                      INT                     ItMax
                  );
template  void  RealRootsOfRealPolynome<REAL16>
                  (
                      ElSTDNS vector<REAL16> &  Sols,
                      const ElPolynome<REAL16>  &aPol,
                      REAL16                    tol,
                      INT                     ItMax
                  );













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
