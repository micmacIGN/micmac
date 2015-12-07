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


#ifndef _ELISE_ALGO_GEOM_GR_QDT_IMPLEM_H
#define _ELISE_ALGO_GEOM_GR_QDT_IMPLEM_H

#include "algo_geom/qdt.h"

/*
     QtArc doit etre une classe de Quod-Tree contenant des
     les ElArc<AttrSom,AttrArc> *

     sgr doit etre une structure de sous graphe redefinissant la 
     methode pt()
*/

#define Tpl_OkTopoDefAngle -1

template <class AttrSom,class AttrArc,class QtArc>
         bool  OkTopo
               (
                    ElSom<AttrSom,AttrArc> *       s1,
                    ElSom<AttrSom,AttrArc> *       s2,
                    ElSubGraphe<AttrSom,AttrArc> & sgr,
                    QtArc                        & qt,
                    REAL    epsilon = 0.0,
                    REAL    ang		= Tpl_OkTopoDefAngle
               )
{
    if (s1==s2) return false;
    typedef  ElSom<AttrSom,AttrArc> TSom;
    typedef  ElArc<AttrSom,AttrArc> TArc;

    TArc * a12 = s1->gr().arc_s1s2(*s1,*s2);

     if (a12 && sgr.inA(*a12)) 
        return false;

     Pt2dr p1 = sgr.pt(*s1);
     Pt2dr p2 = sgr.pt(*s2);
     if (p1 == p2) return false;

     SegComp s12(p1,p2);

	 std::set<TArc *> sa;
     qt.RVoisins(sa,SegComp(p1,p2),epsilon );


     for(typename std::set<TArc *>::iterator it=sa.begin(); it!=sa.end(); it++)
      {
            TArc * arc = *it;
            if (sgr.inA(*arc))
            {
                 TSom * s_in = 0;
                 TSom * s_out = 0;
                 if (
                           (s1 == &(arc->s1()))
                       ||  (s2 == &(arc->s1()))
                    )
                 {
                     s_in = &(arc->s1());
                     s_out = &(arc->s2());
                 }
                 if (
                           (s1 == &(arc->s2()))
                       ||  (s2 == &(arc->s2()))
                    )
                 {
                     s_in = &(arc->s2());
                     s_out = &(arc->s1());
                 }
                 if (! s_in) return false;
                 Pt2dr pout = sgr.pt(*s_out);
                 if (s12.in_bande(pout,SegComp::seg))
                 {
                      Pt2dr pin = sgr.pt(*s_in);
                      if (angle_de_droite_nor(s12.tangente(),pin-pout) < ang)
                         return false;
                 }
            }
      }

     return true;
}



/*
template <class AttrSom,class AttrArc,class SubGrapheGeom>
         class ElGrQdt : public ElGraphe<AttrSom,AttrArc>
{
     public :

           ElGrQdt();
      
     private :

            typedef ElGrQdt<AttrSom,AttrArc,SubGrapheGeom> TyGr;

            ElQT<TSom *,Pt2dr,TyGr &>  mQtSom;


            Pt2dr operator()(TSom * aSomPtr) {return mSGG.pt(*aSomPtr);

            SubGrapheGeom & mSGG;

            virtual void OnNewSom(TSom &);
            virtual void OnNewArc(TArc &);
            virtual void OnKillSom(TSom &){}
            virtual void OnKillArc(TArc &){}

};
*/



#endif //  _ELISE_ALGO_GEOM_GR_QDT_IMPLEM_H





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
