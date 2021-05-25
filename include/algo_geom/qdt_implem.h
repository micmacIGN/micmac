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


#ifndef _ELISE_ALGO_GEOM_QDT_IMPLEM_H
#define _ELISE_ALGO_GEOM_QDT_IMPLEM_H

#include "algo_geom/qdt.h"
#include "ext_stl/elslist.h"

#if (!ELISE_windows)
#include "qdt_insertobj.h"
#endif

#include <set>

/*************************************************************/


template <class Obj> class cTplSetResRVoisin : public cTplResRVoisin<Obj>
{ 
    public :
         
      cTplSetResRVoisin(std::set<Obj> & aSet) : mSet(aSet) {}
    virtual ~cTplSetResRVoisin() {}

    private :
       void Add(const Obj & anObj) {mSet.insert(anObj);}
       std::set<Obj> & mSet;
};

template <class Obj,class Prim,class FPrim>  
         class ElQTArbre : 
               public ElClassQT<Obj,Prim,FPrim>, 
               public Mcheck
{
     public :

          virtual ~ElQTArbre();

          virtual ElQTArbre * insert
                  (
                       typename ElClassQT<Obj,Prim,FPrim>::ArgRequette &,
                       const Obj &        ,
                       const Prim &       ,
                       const Box2dr &     
                   ) = 0;

          virtual void RVoisins
                  (
                       typename ElClassQT<Obj,Prim,FPrim>::ArgRequette          &,
                       cTplResRVoisin<Obj>  &,
                       const Box2dr         &,
                       const ElQTRegionPlan &,
                       REAL                 d2
                  ) =0;

          virtual ElQTArbre *  remove
                  (
                       typename ElClassQT<Obj,Prim,FPrim>::ArgRequette &,
                       const Obj &        ,
                       const Prim &       ,
                       const Box2dr &     
                  ) = 0;

          virtual void  clear(ElQT<Obj,Prim,FPrim> & qt) = 0;

};


template <class Obj,class Prim,class FPrim>
         class ElQTBranche : 
               public ElQTArbre<Obj,Prim,FPrim>
{
      public :

          virtual ~ElQTBranche();
          ElQTBranche();

          virtual ElQTArbre<Obj,Prim,FPrim> * insert
                  (
                       typename ElClassQT<Obj,Prim,FPrim>::ArgRequette &,
                       const Obj &        ,
                       const Prim &       ,
                       const Box2dr &     
                   );

          virtual void RVoisins
                  (
                       typename ElClassQT<Obj,Prim,FPrim>::ArgRequette &,
                       cTplResRVoisin<Obj>  &,
                       const Box2dr         &,
                       const ElQTRegionPlan &,
                       REAL                 d2
                  );

          virtual ElQTArbre<Obj,Prim,FPrim> *    remove
                  (
                       typename ElClassQT<Obj,Prim,FPrim>::ArgRequette &,
                       const Obj &        ,
                       const Prim &       ,
                       const Box2dr &     
                  );
          virtual void  clear(ElQT<Obj,Prim,FPrim> & qt);


      private :
         ElQTArbre<Obj,Prim,FPrim> *  _fils[4];
         ElQTArbre<Obj,Prim,FPrim> * FilsKInit(INT k)
         {
            if (! _fils[k]) 
               _fils[k] = new ElQTFeuille<Obj,Prim,FPrim>;
            return _fils[k];
         }

};


template <class Obj,class Prim,class FPrim>
         class ElQTFeuille : 
               public ElQTArbre<Obj,Prim,FPrim>
{
      public :

          ElQTFeuille();
          virtual ~ElQTFeuille();


          virtual ElQTArbre<Obj,Prim,FPrim> * insert
                  (
                       typename ElClassQT<Obj,Prim,FPrim>::ArgRequette &,
                       const Obj &        ,
                       const Prim &       ,
                       const Box2dr &     
                   );

          virtual void RVoisins
                  (
                       typename ElClassQT<Obj,Prim,FPrim>::ArgRequette &,
                       cTplResRVoisin<Obj>  &,
                       const Box2dr         &,
                       const ElQTRegionPlan &,
                       REAL                 d2
                 );

          virtual ElQTArbre<Obj,Prim,FPrim> *    remove
                  (
                       typename ElClassQT<Obj,Prim,FPrim>::ArgRequette &,
                       const Obj &        ,
                       const Prim &       ,
                       const Box2dr &     
                  );
          virtual void  clear(ElQT<Obj,Prim,FPrim> & qt);


      private :
         typedef typename ElSlist<Obj>::iterator  liter;

         INT       _nb_obj;               
         ElSlist<Obj> _l_obj;
	

};

/**********************************************/
/*                                            */
/* Fonction InsertObj dans  qdt_insertobj.h	  */
/* pour compilation sous windows              */
/*                                            */
/**********************************************/
template <class Obj>void InsertObj(std::set<Obj> & aSet,const Obj & anObj);


/**********************************************/
/*                                            */
/*           ElQTArbre                        */
/*                                            */
/**********************************************/

template <class Obj,class Prim,class FPrim> 
         ElQTArbre<Obj,Prim,FPrim>::~ElQTArbre() 
{
}

/**********************************************/
/*                                            */
/*           ElQTBranche                      */
/*                                            */
/**********************************************/

template <class Obj,class Prim,class FPrim>
         ElQTBranche<Obj,Prim,FPrim>::ElQTBranche()
{
    for (INT k= 0; k< 4 ; k++)
        _fils[k] = 0;
}


template <class Obj,class Prim,class FPrim>
         ElQTBranche<Obj,Prim,FPrim>::~ElQTBranche()
{
    for (INT k= 0; k< 4 ; k++)
        if ( _fils[k])
           delete _fils[k];
}

template <class Obj,class Prim,class FPrim>
          ElQTArbre<Obj,Prim,FPrim> * ElQTBranche<Obj,Prim,FPrim>::insert
          (
                       typename ElClassQT<Obj,Prim,FPrim>::ArgRequette &     arg,
                       const Obj &       obj,
                       const Prim &      prim,
                       const Box2dr &    box 
           )
{
     Box2dr qb[4];
     box.QSplit(qb);

     for (INT k= 0; k< 4 ; k++)
        if (qb[k].Intersecte(prim))
        {
            FilsKInit(k);
            _fils[k] = _fils[k]->insert(arg,obj,prim,qb[k]);
        }
     return this;
}

template <class Obj,class Prim,class FPrim>
          ElQTArbre<Obj,Prim,FPrim> * ElQTBranche<Obj,Prim,FPrim>::remove
          (
                typename ElClassQT<Obj,Prim,FPrim>::ArgRequette &     arg,
                const Obj &       obj,
                const Prim &      prim,
                const Box2dr &    box 
          )
{
     Box2dr qb[4];
     box.QSplit(qb);

     for (INT k= 0; k< 4 ; k++)
        if (_fils[k] && qb[k].Intersecte(prim))
            _fils[k] = _fils[k]->remove(arg,obj,prim,qb[k]);
     return this;
}

template <class Obj,class Prim,class FPrim> 
         void  ElQTBranche<Obj,Prim,FPrim>::clear(ElQT<Obj,Prim,FPrim> & qt)
{

     for (INT k= 0; k< 4 ; k++)
        if (_fils[k])
            _fils[k]->clear(qt);
}





template <class Obj,class Prim,class FPrim>
          void ElQTBranche<Obj,Prim,FPrim>::RVoisins
               (
                       typename ElClassQT<Obj,Prim,FPrim>::ArgRequette & arg,
                       cTplResRVoisin<Obj>             &    TheSet,
                       const Box2dr         &    box ,
                       const ElQTRegionPlan &    reg,
                       REAL                      d2
               )
{
     Box2dr qb[4];
     box.QSplit(qb);

     for (INT k= 0; k< 4 ; k++)
        if (_fils[k] && (reg.D2(qb[k]) < d2))
            _fils[k]->RVoisins(arg,TheSet,qb[k],reg,d2);

}

/**********************************************/
/*                                            */
/*           ElQTFeuille                      */
/*                                            */
/**********************************************/

template <class Obj,class Prim,class FPrim> 
         ElQTFeuille<Obj,Prim,FPrim>::ElQTFeuille() :
              _nb_obj (0),
              _l_obj  ()
{
}

template <class Obj,class Prim,class FPrim> 
         ElQTFeuille<Obj,Prim,FPrim>::~ElQTFeuille() 
{
}

template <class Obj,class Prim,class FPrim>
          ElQTArbre<Obj,Prim,FPrim> * ElQTFeuille<Obj,Prim,FPrim>::insert
          (
                  typename ElClassQT<Obj,Prim,FPrim>::ArgRequette & arg,
                  const Obj &       obj,
                  const Prim &         ,
                  const Box2dr &    box
           )
{
     
    _nb_obj ++;
    _l_obj.push_front(obj,arg._qt.Reserve());

    if (
            (_nb_obj <= arg._qt.NbObjMax())
         || (box.hauteur() <= arg._qt.SzMin())
       )
       return this;

    ElQTBranche<Obj,Prim,FPrim> * br = new ElQTBranche<Obj,Prim,FPrim>;
    for (liter it = _l_obj.begin(); it != _l_obj.end() ; it++)
        br->insert(arg,*it,arg._qt.GPrim(*it),box);
    _l_obj.clear(arg._qt.Reserve());
    delete this;
    return br;
}


template <class Obj,class Prim,class FPrim>
          ElQTArbre<Obj,Prim,FPrim> * ElQTFeuille<Obj,Prim,FPrim>::remove
          (
                  typename ElClassQT<Obj,Prim,FPrim>::ArgRequette & arg,
                  const Obj &       obj,
                  const Prim &         ,
                  const Box2dr &    
           )
{
     
    _nb_obj -= _l_obj.remove(obj,arg._qt.Reserve());
     return this;
}

template <class Obj,class Prim,class FPrim>
          void ElQTFeuille<Obj,Prim,FPrim>::clear(ElQT<Obj,Prim,FPrim> & qt)
{
    _l_obj.clear(qt.Reserve());
}




template <class Obj,class Prim,class FPrim>
          void ElQTFeuille<Obj,Prim,FPrim>::RVoisins
               (
                       typename ElClassQT<Obj,Prim,FPrim>::ArgRequette & arg,
                       cTplResRVoisin<Obj>             &    TheSet,
                       const Box2dr         &    b   ,
                       const ElQTRegionPlan &    reg,
                       REAL                      d2
                )
{

    for (liter it = _l_obj.begin(); it != _l_obj.end() ; it++)
    {
        if (reg.D2(arg._qt.GPrim(*it)) <= d2)
           TheSet.Add(*it);
    }
}
/* */

/**********************************************/
/*                                            */
/*           ElQT                             */
/*                                            */
/**********************************************/

template <class Obj,class Prim,class FPrim> ElQT<Obj,Prim,FPrim>::~ElQT()
{
     delete _racine;
}

// OK
template <class Obj,class Prim,class FPrim> 
          ElQT<Obj,Prim,FPrim>::ElQT
          (
                const FPrim & FPRIM,
                Box2dr        BOX,
                INT           NBOBJMAX,
                REAL          SzMin
          ) :
          NewElQdtGen(BOX,NBOBJMAX,SzMin),
          _fprim    (FPRIM),
          _racine    (new  ElQTFeuille<Obj,Prim,FPrim>)
{
}

template <class Obj,class Prim,class FPrim> 
         bool ElQT<Obj,Prim,FPrim>::insert(const Obj & obj,bool svp)
{
    Prim p (_fprim(obj));
    bool inside = _box.Include(p);

    if (! inside)
    {
        ELISE_ASSERT(svp,"insert inq ElQT, obj out of Box");
        return false;
    }


    typename ElClassQT<Obj,Prim,FPrim>::ArgRequette arg(*this);
    _racine = _racine->insert(arg,obj,p,_box);
    return true;
}

// PAS OK
template <class Obj,class Prim,class FPrim> 
         void ElQT<Obj,Prim,FPrim>::RVoisins
              (
                       cTplResRVoisin<Obj>             & TheSet,
                       const ElQTRegionPlan & reg,
                       REAL                   d
              )
{
    typename ElClassQT<Obj,Prim,FPrim>::ArgRequette arg(*this);
    _racine->RVoisins(arg,TheSet,_box,reg,ElSquare(d));
}


template <class Obj,class Prim,class FPrim>
         void ElQT<Obj,Prim,FPrim>::RVoisins 
              (
                       cTplResRVoisin<Obj> & aRes,
                       Pt2dr                  pt,
                       REAL                   d
              )
{
    RVoisins(aRes,ElQTRegPt(pt),d);
}

template <class Obj,class Prim,class FPrim>
         void ElQT<Obj,Prim,FPrim>::RVoisins 
              (
                       ElSTDNS set<Obj>             & TheSet,
                       Pt2dr                  pt,
                       REAL                   d
              )
{
    cTplSetResRVoisin<Obj>  aRes(TheSet);
    RVoisins(aRes,pt,d);
}




template <class Obj,class Prim,class FPrim>
         void ElQT<Obj,Prim,FPrim>::RVoisins 
              (
                       cTplResRVoisin<Obj> & aRes,
                       Box2dr                  pt,
                       REAL                   d
              )
{
    RVoisins(aRes,ElQTRegBox(pt),d);
}

template <class Obj,class Prim,class FPrim>
         void ElQT<Obj,Prim,FPrim>::RVoisins 
              (
                       ElSTDNS set<Obj>             & TheSet,
                       Box2dr                 box,
                       REAL                   d
              )
{
    cTplSetResRVoisin<Obj>  aRes(TheSet);
    RVoisins(aRes,box,d);
}





template <class Obj,class Prim,class FPrim>
         void ElQT<Obj,Prim,FPrim>::RVoisins 
              (
                       cTplResRVoisin<Obj> & aRes,
                       Seg2d                  pt,
                       REAL                   d
              )
{
    RVoisins(aRes,ElQTRegSeg(pt),d);
}

template <class Obj,class Prim,class FPrim>
         void ElQT<Obj,Prim,FPrim>::RVoisins 
              (
                       ElSTDNS set<Obj>             & TheSet,
                       Seg2d                  seg,
                       REAL                   d
              )
{
    cTplSetResRVoisin<Obj>  aRes(TheSet);
    RVoisins(aRes,seg,d);
}

/*
    RVoisins(TheSet,ElQTRegSeg(seg),d);
*/




template <class Obj,class Prim,class FPrim>
         void ElQT<Obj,Prim,FPrim>::remove(const Obj & obj)
{
    typename ElClassQT<Obj,Prim,FPrim>::ArgRequette arg(*this);
    _racine = _racine->remove(arg,obj,_fprim(obj),_box);
}


template <class Obj,class Prim,class FPrim>
         void ElQT<Obj,Prim,FPrim>::clear()
{
    _racine->clear(*this);
}


template <class Obj> 
class cPairPtrDist
{
    public :
         cPairPtrDist (const Obj * anObj,double aDist) :
             mObj  (anObj),
             mDist (aDist)
         {
         }

         const Obj *  mObj;
         double mDist;
};

template <class Obj> class cCmpPairPtrDist
{
    public :
         bool operator () (
                           const cPairPtrDist<Obj> & aP1,
                           const cPairPtrDist<Obj> & aP2
                         )
         {
               if (aP1.mDist < aP2.mDist) return true;
               if (aP1.mDist > aP2.mDist) return false;
               return (aP1.mObj<aP2.mObj);
         }
};




template <class Obj,class Prim,class FPrim>
         void  ElQT<Obj,Prim,FPrim>::KPPVois
               (
                              std::list<Obj> & aRes,
                              const ElQTRegionPlan & aReg,
                              int aNb,
                              double aDistInit,
                              double aFact,
                              int aNbMaxTest
               )
{
    std::set<Obj> aSetLarg;
    cTplSetResRVoisin<Obj>  aTSRV(aSetLarg);

    int aNbTest = 0;
    while ((int(aSetLarg.size())<aNb) && (aNbTest<aNbMaxTest))
    {
        aNbTest++;
        aSetLarg.clear();
        RVoisins(aTSRV,aReg,aDistInit);
        aDistInit *= aFact;
    }
    std::vector<cPairPtrDist<Obj> > aVP;

    for
    (
         typename std::set<Obj>::iterator itO = aSetLarg.begin();
         itO != aSetLarg.end();
         itO++
    )
    {
       aVP.push_back
       (
            cPairPtrDist<Obj>
            (
                &(*itO),
                aReg.D2(_fprim(*itO))
            )
       );
    }

    cCmpPairPtrDist<Obj> aCmp;
    std::sort(aVP.begin(),aVP.end(),aCmp);
    int aNbRes = ElMin(int(aVP.size()),aNb);
    aRes.clear();
    for (int aK=0 ; aK<aNbRes ; aK++)
       aRes.push_back(*(aVP[aK].mObj));
}



template <class Obj,class Prim,class FPrim>  
         std::list<Obj> ElQT<Obj,Prim,FPrim>::KPPVois
                        (
                              Pt2dr aP,
                              int aNb,
                              double aDistInit,
                              double aFact,
                              int aNbMaxTest
                        )
{
    std::list<Obj> aRes;

    KPPVois(aRes,ElQTRegPt(aP),aNb,aDistInit,aFact,aNbMaxTest);

    return aRes;
}

template <class Obj,class Prim,class FPrim>
          cTplValGesInit<Obj>  ElQT<Obj,Prim,FPrim>::NearestObjSvp
          (
                Pt2dr aP,
                double aDistInit,
                double aDistMax
          )
{
    int aNbMax = round_up(log2(aDistMax/aDistInit));
    aDistInit = aDistMax/pow(2.0,aNbMax);

    std::list<Obj> aLObj = KPPVois(aP,1,aDistInit,2.0,aNbMax);

    cTplValGesInit<Obj> aRes;
    if (!aLObj.empty())
       aRes.SetVal(*(aLObj.begin()));
    return aRes;
}



template <class Obj,class Prim,class FPrim> 
   Obj   ElQT<Obj,Prim,FPrim>::NearestObj
          (
                Pt2dr aP,
                double aDistInit,
                double aDistMax
          )
{
/*
    int aNbMax = round_up(log2(aDistMax/aDistInit));
    aDistInit = aDistMax/pow(2.0,aNbMax);

    std::list<Obj> aLObj = KPPVois(aP,1,aDistInit,2.0,aNbMax);
*/
    
    cTplValGesInit<Obj>  aRes = NearestObjSvp(aP,aDistInit,aDistMax);
    ELISE_ASSERT(aRes.IsInit(),"ElQT<Obj,Prim,FPrim>::NearestObj");
    return aRes.Val();
}


//  Prerequis pour param
//   bool operator()(const Type & aPI1,const Type & aPI2) ; => Indique si PI1 es meilleurs que PI2
//   Pt2dr operator()(const Type * aPI) ;   
//   bool IsSelected(const cIntTieTriInterest & aPI) ;    
//   void SetSelected(cIntTieTriInterest & aPI,bool aVal) ;


template <class Type,class cParam> class   cTplFiltrageSpatial
{
    public :

  // === For case filter spatial in TiepTri : aVType is vector of interest point, aParam is a comparator by FastQuality ===
  cTplFiltrageSpatial(std::vector<Type> & aVType,cParam  & aParam,double aSeuilDist)
  {
     if (aVType.size() <=1 )
        return;

// === sort par comparator aParam => sort list point d'interet dans la priorité de FastQuality ===
     std::sort(aVType.begin(),aVType.end(),aParam);

     Pt2dr aP0 = aParam(&(aVType[0]));
     Pt2dr aP1 = aP0;

// === Define an Quad-Tri ElQT structure : range (object space (min,max)) ===
     for (int aK=1 ; aK<int(aVType.size()) ; aK++)
     {
         Pt2dr aPk = aParam(&(aVType[aK]));
         aP0 = Inf(aP0,aPk);    // aP0 store min value of vector
         aP1 = Sup(aP1,aPk);    // aP1 store max value of vector
     }
     Pt2dr aSz = aP1-aP0;   // range max-min
     double aLong = ElMax(aSz.x,aSz.y);
     double aLarg = ElMin(aSz.x,aSz.y);

// === If aLong=0 => input point set has just 1 element => return and do nothing ===
     if (aLong==0)
     {
        Type aV0 = aVType[0];
        aVType.clear();
        aVType.push_back(aV0);

        return;
     }

     double aRab = aLong / 1e4;
     Pt2dr aPRab(aRab,aRab);
     double aSurfElem = (aLong * ElMax(aLarg,aLong/20.0)) / aVType.size()  ;


// === Create a QuadTri structure : 10 is NbObjectMax, sqrt(aSurfElem) *2 is SzMin ===
     ElQT<Type *,Pt2dr,cParam>  aQt(aParam,Box2dr(aP0-aPRab,aP1+aPRab),10, sqrt(aSurfElem) *2 );


     // "insert" tout les pt interet dans ElQT, initialization = "selected"
     for (int aK=0 ; aK<int(aVType.size()) ; aK++)
     {
         // "insert" marche toujours parcque le "box" dans ElQT est la region Box de (Min, Max) coordonne du point d'interet
         aQt.insert(&aVType[aK]);
         aParam.SetSelected(aVType[aK],true);
     }


// ==== Filter process : process element by element in sorted list aVType ===
     std::vector<Type> aRes;
     for (int aK=0 ; aK<int(aVType.size()) ; aK++)
     {
         // consider un point d'interet (dans l'ordre de fastquality
         if (aParam.IsSelected(aVType[aK]))
         {
             aRes.push_back(aVType[aK]);
             std::set<Type *> aSet;
             // recuperer tout les point dans rayon aSeuilDist autour de point aVType[aK], le stocker dans aSet
             aQt.RVoisins(aSet,aParam(&(aVType[aK])),aSeuilDist);

             for (typename std::set<Type *>::iterator itS=aSet.begin(); itS!=aSet.end() ; itS++)
             {
                 // De-selectioner tout les point viens de recuperer
                  aParam.SetSelected(**itS,false);
             }
         }
     }

     aVType = aRes;
  }
};

#endif //  _ELISE_ALGO_GEOM_QDT_IMPLEM_H

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
