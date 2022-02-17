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

#ifndef _ELISE_GRAPHE_ALGO_DUAL
#define _ELISE_GRAPHE_ALGO_DUAL
#include "graphes/algo_planarite.h"
#include <map>


template <class  Graphe, class GrDual> class ElMakeDual
{
     public :

        //======================
        // Section typedef 
        //======================

        typedef ElTyName Graphe::ASom                 AttrSom;
        typedef ElTyName Graphe::AArc                 AttrArc;
        typedef ElSubGraphe<AttrSom,AttrArc>          SubGraphe;
        typedef ElArc<AttrSom,AttrArc>                TArc;
        typedef ElSom<AttrSom,AttrArc>                TSom;             

        typedef ElTyName GrDual::ASom                 DASom;
        typedef ElTyName GrDual::AArc                 DAArc;
        typedef ElSom<DASom,DAArc>                    DTSom;
        typedef ElArc<DASom,DAArc>                    DTArc;

        typedef map<TArc *,INT>                       MapArc;



        //====================================================
        // Section Methode a (re)definir  par l'utilisateur 
        //====================================================

        virtual bool  DM_OkFace(ElSubFilo<TArc *> )
        {
            return true;
        }

        virtual  DAArc  DM_CreateAttrArcDualInit(DTSom & , DTSom & )
        {
             return DAArc();
        }
        virtual  DAArc  DM_CreateAttrArcDualRec(DAArc & attr,DTSom & , DTSom & )
        {
             return attr;
        }
        virtual void DM_CumulAttrArcDual(DTArc & ,TArc &)
        {
        }



        virtual  DASom  DM_CreateAttrFace(ElSubFilo<TArc *>)
        {
             return DASom();
        }



        void make_dual(Graphe &  gr,SubGraphe & sub,GrDual & dual)
        {
             ElPartition<TArc *> FArcs;
             ElFilo<DTSom *>     FDual;
             MapArc              FaceD;


              all_face_trigo(gr,sub,FArcs);
             for (INT Kf =0 ; Kf<FArcs.nb(); Kf++)
             {
                 ElSubFilo<TArc *>  fa = FArcs[Kf];

                 if (DM_OkFace(fa))
                 {
                    DTSom & f = dual.new_som(DM_CreateAttrFace(fa));
                    FDual.pushlast(&f);
                    for (INT ka = 0 ; ka<fa.nb() ; ka++)
                    {
                         //  FaceG[fa[ka]] = Kf; => acces implicite
                         FaceD[&(fa[ka]->arc_rec())] = Kf;
                    }
                 }
                 else
                 {
                    FDual.pushlast(0);
                 }
             }

             for (INT Kf =0 ; Kf<FArcs.nb(); Kf++)
             {
                  DTSom * f1 = FDual[Kf];
                  if (f1 != 0)
                  {
                      ElSubFilo<TArc *>  fa = FArcs[Kf];
                      for (INT ka = 0 ; ka<fa.nb() ; ka++)
                      {
                          TArc * arcInit = fa[ka];
                          MapArc::iterator ItD = FaceD.find(arcInit);
                          if (ItD != FaceD.end())
                          {
                             DTSom * f2 = FDual[ItD->second];
                             if (f1 != f2)
                             {
                                DTArc * arcD = dual.arc_s1s2(*f1,*f2);
                                if (! arcD)
                                {
                                    DAArc a12 = DM_CreateAttrArcDualInit(*f1,*f2);
                                    DAArc a21 = DM_CreateAttrArcDualRec(a12,*f2,*f1);
                                    DTArc & NarcD = dual.add_arc(*f1,*f2,a12,a21);
                                    arcD = & NarcD;
                                }
                                DM_CumulAttrArcDual(*arcD,*arcInit);
                             }
                          }
                      }
                  }
             }
              
        }
};


#endif // _ELISE_GRAPHE_ALGO_DUAL










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
