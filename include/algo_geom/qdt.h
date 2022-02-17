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


#ifndef _ELISE_ALGO_GEOM_QDT_H
#define _ELISE_ALGO_GEOM_QDT_H

/*
    template <class Obj,class Prim,class FPrim> class ElClassQT;

      Obj : le type des objets contenu dans la quod tree
      Prim : le type de primitive geometrique contenu 
      FPrim : un  type telq que FPrim(Obj) renvoie une Prim


      template <class Type>  bool Box2d<Type>::Intersecte(const class SegComp & seg) const
         Necessaire pour ajouter l'element dans le Qdt

      template <class Type>  bool Box2d<Type>::Include(const class SegComp & seg) const
         Necessaire pour savoir si l'element est inclus dans le qdt

      virtual REAL ElQTRegionPlan::D2(const Pt2dr &)  const = 0;

      La fonction D2 doit etre redefinie sur les primitives pour lequelles
      on veut acceder a R Voisin
*/


template <class Obj,class Prim,class FPrim>  class ElClassQT;
template <class Obj,class Prim,class FPrim>  class ElQTArbre;
template <class Obj,class Prim,class FPrim>  class ElQTBranche;
template <class Obj,class Prim,class FPrim>  class ElQTFeuille;
template <class Obj,class Prim,class FPrim>  class ElQT;

template <class Obj,class Prim,class FPrim>
         class ElClassQT
{
    public :


       class  ArgRequette
       {
           public :

              ArgRequette(ElQT<Obj,Prim,FPrim> & qt) :
                   _qt (qt)
              {
              }

              ElQT<Obj,Prim,FPrim> & _qt;
       };


    private :
};

template <class Obj> class cTplResRVoisin
{
    public :
       virtual void Add(const Obj &) = 0;
    virtual ~cTplResRVoisin() {}

};

template <class Obj> class cVecTplResRVoisin : public cTplResRVoisin<Obj> ,
                                               public std::vector<Obj>
{
    public :
       virtual void Add(const Obj & anObj) {this->push_back(anObj);}
       virtual ~cVecTplResRVoisin() {}

};

template <class Obj,class Prim,class FPrim> 
          class ElQT : public   ElClassQT<Obj,Prim,FPrim>,
                       public   NewElQdtGen
{
    public :

            ElQT
            (
                const FPrim & fprim,
                Box2dr        box,
                INT           NbObjMax,
                REAL          SzMin
            );

                // insert : renvoie true si l'objet est effectivement inclu
                // la box du QT, cepdendant si svp vaut false  et 
                // que l'objet est hors box une erreur  est lancee

            bool insert(const Obj &,bool svp = false);

            std::list<Obj> KPPVois(Pt2dr,int aNb,double aDistInit,double aFact=2.0,int aNbMax=10);


            Obj  NearestObj(Pt2dr,double aDistInit,double aDistMax); // Erreur si vide
            cTplValGesInit<Obj>  NearestObjSvp(Pt2dr,double aDistInit,double aDistMax); // peut etre vide

            void RVoisins(ElSTDNS set<Obj> &,Pt2dr   pt,REAL d);
            void RVoisins(ElSTDNS set<Obj> &,Seg2d   pt,REAL d);
            void RVoisins(ElSTDNS set<Obj> &,Box2dr  pt,REAL d);

            void RVoisins(cTplResRVoisin<Obj> &,Pt2dr   pt,REAL d);
            void RVoisins(cTplResRVoisin<Obj> &,Seg2d    pt,REAL d);
            void RVoisins(cTplResRVoisin<Obj> &,Box2dr   pt,REAL d);

            void remove(const Obj & obj);
            void clear();

            virtual ~ElQT();


        // =============================================================
        // =============================================================
        // =============================================================
        // =============================================================

            Prim  GPrim(const Obj & obj) { return _fprim(obj);}
            ElSlist<Obj> * Reserve() {return & _reserve;}


    private :

           ElSlist<Obj>                _reserve;
           FPrim                         _fprim;
           ElQTArbre<Obj,Prim,FPrim> *   _racine;

           virtual void RVoisins
                  (
                       cTplResRVoisin<Obj>  &,
                       const ElQTRegionPlan &,
                       REAL                 d
                  );

           void           KPPVois
                          (
                               std::list<Obj> & aRes,
                               const ElQTRegionPlan &,
                               int aNb,
                               double aDistInit,
                               double aFact,
                               int aNbMax
                           );
};

#endif //  _ELISE_ALGO_GEOM_QDT_H



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
