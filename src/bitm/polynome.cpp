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

/*************************************************************/
/*                                                           */
/*     Construction-Destruction etc ..                       */
/*                                                           */
/*************************************************************/

template <class Type> ElPolynome<Type>  
         ElPolynome<Type>::FromRoots(const ElSTDNS vector<Type> & VRoots)
{
   ElPolynome <Type> aRes(El1);
   for (INT aK=0; aK<INT(VRoots.size()) ; aK++)
	   aRes = aRes *  FromRoots(VRoots[aK]);
   return aRes;
}

template <class Type> ElPolynome<Type> 
        ElPolynome<Type>::FromRoots(const Type & aRoot)
{
   return ElPolynome<Type>(-aRoot,El1);
}

template <class Type> ElPolynome<Type> 
        ElPolynome<Type>::FromRoots(const Type & aR1,const Type & aR2)
{
    return FromRoots(aR1) * FromRoots(aR2);
}

template <class Type> ElPolynome<Type> 
	        ElPolynome<Type>::FromRoots
                      (const Type & aR1,const Type & aR2,const Type & aR3)
{
    return FromRoots(aR1,aR2) * FromRoots(aR3);
}

template <class Type> ElPolynome<Type> 
	        ElPolynome<Type>::FromRoots
                      (const Type & aR1,const Type & aR2,const Type & aR3,const Type & aR4)
{
    return FromRoots(aR1,aR2) * FromRoots(aR3,aR4);
}



template <class Type> ElPolynome<Type>::ElPolynome()
{
    _coeff.push_back(El0);
}

template <class Type> ElPolynome<Type>::ElPolynome(const Type & c0)
{
    _coeff.push_back(c0);
}

template <class Type> ElPolynome<Type>::ElPolynome
                      (
                          const Type & c0,
                          const Type & c1
                      )
{
    _coeff.push_back(c0);
    _coeff.push_back(c1);
}

template <class Type> ElPolynome<Type>::ElPolynome
                      (
                          char *,
                          INT degre
                      )
{
    for (INT d=0 ; d<= degre ; d++)
       _coeff.push_back(El0);
}

template <class Type> ElPolynome<Type>::ElPolynome
                      (
                          const Type & c0,
                          const Type & c1,
                          const Type & c2
                      )
{
    _coeff.push_back(c0);
    _coeff.push_back(c1);
    _coeff.push_back(c2);
}

template <class Type>  Type  ElPolynome<Type>::operator()(const Type & x) const
{
    Type Xk = El1;
    Type res = El0;
    for (INT k=0 ; k<(int)_coeff.size(); k++)
    {
        res = res + Xk * _coeff[k];
        Xk  = Xk * x;
    }
    return res;
}

template <class Type>  
         ElPolynome<Type> ElPolynome<Type>::operator *
                           (const ElPolynome<Type> & p2) const
{
   INT d1 = degre();
   INT d2 = p2.degre();
   ElPolynome<Type> res((char *)0,d1+d2);

   for (int k1=0; k1<=d1 ; k1++)
       for (int k2=0; k2<=d2 ; k2++)
           res._coeff[k1+k2] = res._coeff[k1+k2]+_coeff[k1]*p2._coeff[k2];

   return res;
}



template <class Type>  const Type &  ElPolynome<Type>::operator[](INT k) const
{
    ELISE_ASSERT(((k>=0)&&(k<(int)_coeff.size())),"ElPolynome[]");
    return _coeff[k];
}

template <class Type>  Type &  ElPolynome<Type>::operator[](INT k)
{
    ELISE_ASSERT(((k>=0)&&(k<(int)_coeff.size())),"ElPolynome[]");
    return _coeff[k];
}

template <class Type>  Type ElPolynome<Type>::at(INT k) const
{
    if ((k>=0)&&(k<(int)_coeff.size()))
       return _coeff[k];
    else
       return El0;
}

template <class Type>  
         ElPolynome<Type> ElPolynome<Type>::operator +
                           (const ElPolynome<Type> & p2) const
{
   INT d1 = degre();
   INT d2 = p2.degre();
   INT d = ElMax(d1,d2);
   ElPolynome<Type> res((char *)0,d);

   for (int k=0; k<=d ; k++)
           res._coeff[k] = at(k)+p2.at(k);

   return res;
}

template <class Type>  
         ElPolynome<Type> ElPolynome<Type>::operator -
                           (const ElPolynome<Type> & p2) const
{
   INT d1 = degre();
   INT d2 = p2.degre();
   INT d = ElMax(d1,d2);
   ElPolynome<Type> res((char *)0,d);

   for (int k=0; k<=d ; k++)
           res._coeff[k] = at(k)-p2.at(k);

   return res;
}


template <class Type>
         ElPolynome<Type> ElPolynome<Type>::deriv () const
{
   ElPolynome<Type> res(*this);
   res.self_deriv();
   return res;
}                

template <class Type> void ElPolynome<Type>::self_deriv ()
{
    for (INT k=0 ; k<degre() ; k++)
        _coeff[k] = _coeff[k+1] * REAL(k+1);
    _coeff.pop_back();
}

void Reduce(ElPolynome<REAL> & aPol)
{
    while (aPol._coeff.size() && (aPol._coeff[aPol.degre()] == 0))
       aPol._coeff.pop_back();
}

template <class Type>  
         ElPolynome<Type> ElPolynome<Type>::operator *
                           (const Type & v) const
{
   INT d = degre();
   ElPolynome<Type> res((char *)0,d);

   for (int k=0; k<=d ; k++)
           res._coeff[k] = _coeff[k]*v;

   return res;
}


/*
template <class Type>  ElPolynome<Type> operator *
                       (const Type & v,const ElPolynome<Type> & p)
{
   return p*v;
}
*/

template <> const REAL   ElPolynome<REAL >::El0 (0.0);
template <> const REAL   ElPolynome<REAL >::El1 (1.0);
template <> const Pt2dr  ElPolynome<Pt2dr>::El0 (0.0,0.0);
template <> const Pt2dr  ElPolynome<Pt2dr>::El1 (1.0,0.0);


template class ElPolynome<REAL>;  
// template ElPolynome<REAL>  operator * (const REAL &,const  ElPolynome<REAL> &);
template class ElPolynome<Pt2dr>;  


template <> const float   ElPolynome<float >::El0 (0.0);
template <> const float   ElPolynome<float >::El1 (1.0);
template class ElPolynome<float>;  


template <> const REAL16   ElPolynome<REAL16 >::El0 (0.0);
template <> const REAL16   ElPolynome<REAL16 >::El1 (1.0);
template class ElPolynome<REAL16>;  


ElPolynome<double> LeasSqFit(vector<Pt2dr> aVSamples,int aDeg,const std::vector<double> * aVPds)
{
   if (aVPds)
   {
      ELISE_ASSERT(aVPds->size()==aVSamples.size(),"Incoh size Pde/aVSamples in LeasSqFit");
   }

   if (aDeg==-1)
      aDeg = aVSamples.size()-1;
   int aNbVar = aDeg +1 ;

   ELISE_ASSERT
   (
        aNbVar<=int(aVSamples.size()),
        "LeasSqFit Poly, degree too low"
   );

   L2SysSurResol aSys(aNbVar);

   for (int aKS=0 ; aKS<int(aVSamples.size()) ; aKS++)
   {
       double aPds = aVPds ? (*aVPds)[aKS] : 1.0;
       std::vector<double> aVPowX;
       aVPowX.push_back(1.0);
         
       for (int aD=1 ; aD<= aDeg ; aD++)
       {
           aVPowX.push_back(aVPowX.back()*aVSamples[aKS].x);
       }

       aSys.AddEquation(aPds,VData(aVPowX),aVSamples[aKS].y);
   }

   ElPolynome<double>  aRes((char *)0,aDeg);
   Im1D_REAL8  aSol = aSys.Solve((bool *)0);

   for (int aD=0 ; aD<=int(aDeg) ; aD++)
   {
       aRes[aD] = aSol.data()[aD];
   }

/*
   std::cout <<  "FIT Deg=" << aDeg << " NbObs = " << aVSamples.size() <<"\n";

   for (int aKS=0 ; aKS<int(aVSamples.size()) ; aKS++)
       std::cout << " TEST " << aVSamples[aKS].x
                 << " Y " << aVSamples[aKS].y -  aRes(aVSamples[aKS].x)
                 // << " Y " << aRes(aVSamples[aKS].x)
                 << "\n";

*/
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
