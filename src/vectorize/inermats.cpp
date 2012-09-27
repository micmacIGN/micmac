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

template <class Type> 
Mat_Inertie<Type>::Mat_Inertie() :
   _s   (0),
   _s1  (Type::El0()),
   _s2  (Type::El0()),
   _s11 (0),
   _s12 (0),
   _s22 (0)
{
}                  



template <class Type> REAL square_dist_droite(const SegComp & seg, const Mat_Inertie<Type> & m )
{
    return 
                m.s()  *  seg.c()            *  seg.c()
        +  2 * m.s1()  *  seg.c()            *  seg.normale().x  
        +  2 * m.s2()  *  seg.c()            *  seg.normale().y 
        +      m.s11() *  seg.normale().x    *  seg.normale().x
        +  2 * m.s12() *  seg.normale().x    *  seg.normale().y
        +      m.s22() *  seg.normale().y    *  seg.normale().y;
}



template <class Type> Mat_Inertie<Type> Mat_Inertie<Type>::operator - (const Mat_Inertie<Type> & m2) const
{
    return Mat_Inertie<Type>
           (
               _s   - m2._s,
               _s1  - m2._s1,
               _s2  - m2._s2,
               _s11 - m2._s11,
               _s12 - m2._s12,
               _s22 - m2._s22
           );
}


template <class Type> void Mat_Inertie<Type>::operator += (const Mat_Inertie<Type> & m2) 
{
    _s   += m2._s;
    _s1  += m2._s1;
    _s2  += m2._s2;
    _s11 += m2._s11;
    _s12 += m2._s12;
    _s22 += m2._s22 ;
}

static REAL  RMI(REAL  aVA,REAL  aVB,REAL epsilon)
{
   return sqrt(ElMax(0.0,aVA/ElMax(aVB,epsilon)));
}

template <class Type>   typename Type::TypeReel::TypeEff   
                        Mat_Inertie<Type>::V2toV1
                       (const typename Type::TypeReel::TypeEff & aV2,REAL e)
{
   Mat_Inertie<typename Type::TypeReel>  aMN = normalize();
   return aMN.s1() + (aV2-aMN.s2()) * RMI(aMN.s11(),aMN.s22(),e);
}

template <class Type>   typename Type::TypeReel::TypeEff   
                        Mat_Inertie<Type>::V1toV2
                       (const typename Type::TypeReel::TypeEff & aV1,REAL e)
{
   Mat_Inertie<typename Type::TypeReel>  aMN = normalize();
   return aMN.s2() + (aV1-aMN.s1()) *  RMI(aMN.s22(),aMN.s11(),e);
}



template <class Type> Seg2d  seg_mean_square(const Mat_Inertie<Type> & Minitiale, REAL norm)
{

   Mat_Inertie<ElTypeName_NotMSW Type::TypeReel> m = Minitiale.normalize();

// si mat inert circulaire, alors n'importe
// quelle droite passant par le cdg

     Pt2dr cdg(m.s1(),m.s2());

     if ((m.s11()==m.s22()) && (m.s12() == 0.0))
     {
           return Seg2d(cdg,cdg+Pt2dr(norm,norm));
     }

/*
   Sinon, les sxx,syy sxy sont exprime dans le repere du cdg,
   on calcule donc d'abord la droite origine minimisant l'emq
   puis on la translate

   Soit une droite origine d'angle Q, 
   si dans square_dist_droite(const SegComp & seg)  on pose
   normale =  (-sin(Q),cos(Q)) on voit que la dist D(Q) :


      D(Q) = 
               (Sxx+Syy)/2
     +         (Syy-Sxx)/2 * cos(2Q)
     -         Sxy         * sin(2Q)


     Soient rho, phi tq :
          rho*(cos(phi),sin(phi)) = ((_syy-_sxx)/2.0,_sxy))

       (ie phi = atan2(_sxy,_syy-_sxx))

    donc
       D(Q) =  (Sxx+Syy)/2 
              + cos(Phi)cos(2Q) -sin(phi)sin(2Q)

        =   cste + cos(Phi-2Q)

        Cette valeur mini pour Phi-2Q = PI

*/

     REAL phi = atan2(m.s12(),(m.s22()-m.s11())/2.0);
     REAL teta = (PI-phi)/2.0;

     return  Seg2d
             (
                cdg,
                cdg+ Pt2dr::FromPolar(norm,teta)
             );
}

template class Mat_Inertie<ElStdTypeScal<INT> >;
template class Mat_Inertie<ElStdTypeScal<REAL> >;
template class Mat_Inertie<Pt2di>;
template class Mat_Inertie<Pt2dr>;

template REAL square_dist_droite(const SegComp &, const IMat_Inertie &);
template REAL square_dist_droite(const SegComp &, const RMat_Inertie &);

template  Seg2d seg_mean_square(const  IMat_Inertie &,REAL norm);
template  Seg2d seg_mean_square(const  RMat_Inertie &,REAL norm);










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
