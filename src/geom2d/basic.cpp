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


void PtsOfSquare(ElFifo<Pt2dr> & pts,Pt2dr p0,Pt2dr p1)
{
    pts.set_circ(true);
    pts.clear();
    Pt2dr H = p1 -p0;
    Pt2dr V = H * Pt2dr(0,1);


    pts.pushlast(p0);
    pts.pushlast(p1);
    pts.pushlast(p1+V);
    pts.pushlast(p1+V-H);

}
 


REAL surf_or_poly(const ElFifo<Pt2dr> & f)
{
     REAL res = 0.0;
     for (INT k=0; k<f.nb(); k++)
         res += f[k]^f[k+1];
     return res /2.0;
}

REAL surf_or_poly(const std::vector<Pt2dr> & f)
{
     INT nb = (int) f.size();
     REAL res = 0.0;

     for (INT k=0; k<nb; k++)
         res += f[k]^f[(k+1)%nb];
     return res /2.0;
}

REAL perimetre_poly(const std::vector<Pt2dr> & f)
{
     INT nb = (INT) f.size();
     REAL res = 0.0;

     for (INT k=0; k<nb; k++)
         res += euclid(f[k],f[(k+1)%nb]);
     return res;
}


template <class TypePile> Pt2dr Tbarrycentre (const TypePile & f)
{
    Pt2dr cdg(0,0);
    REAL s_pds = 0.0;

    ELISE_ASSERT(f.size()!=0,"No point in bary");
    Pt2dr aOri = f[0];

    for (INT k=0; k<(INT)f.size(); k++)
    {
        Pt2dr p0 =f[k]-aOri;
        Pt2dr p1 =f[(k+1)%f.size()]-aOri;
        REAL pds = p0^p1;
        s_pds += pds;
        cdg += (p0+p1) * pds;
    }

    if (s_pds == 0)
    {
       std::cout << "For polyg = " ;
    
       for (INT k=0; k<(INT)f.size(); k++)
       {
          Pt2dr p0 =f[k];
          Pt2dr p1 =f[(k+1)%f.size()];
          if (k==0) std::cout << "P0 : " << p0;
          std::cout << p1-p0 ;
       }
       std::cout << "\n";
       ELISE_ASSERT(false,"barrycentre, surface nulle");
    }
    return  cdg / (3*s_pds) + aOri;
}


Pt2dr barrycentre (const ElFifo<Pt2dr> & f)
{
    return Tbarrycentre(f);
}
Pt2dr barrycentre (const  std::vector<Pt2dr> & f)
{
    return Tbarrycentre(f);
}


template <class TypePile> bool TplPointInPoly(const TypePile & f,Pt2dr pt)
{

     INT nb_inter = 0;
     for (INT k=0; k<(INT)f.size() ; k++)
     {
          Pt2dr q0 = f[k];
          Pt2dr q1 = f[(k+1)%f.size()];
          if ((q0.y>pt.y) != (q1.y >pt.y))
          {
              REAL absc =  (q0.x-pt.x) - (q0.y-pt.y)*((q1.x-q0.x)/(q1.y - q0.y)); 
              if (absc >=0)
                 nb_inter ++;
          }
     }

     return (nb_inter %2) != 0;
}

bool PointInPoly(const ElFifo<Pt2dr> & f,Pt2dr pt)
{
    return TplPointInPoly(f,pt);
}

bool PointInPoly(const std::vector<Pt2dr> & f,Pt2dr pt)
{
    return TplPointInPoly(f,pt);
}







bool PointInterieurPoly(const ElFifo<Pt2dr> & f,Pt2dr pt,REAL d)
{
     if (! PointInPoly(f,pt))
        return false;

     REAL d2 = ElSquare(d);

     for (INT k=0; k<f.nb() ; k++)
     {
          Pt2dr q0 = f[k];
          Pt2dr q1 = f[k+1];
          if ((q0!=q1) && (SegComp(q0,q1).square_dist_seg(pt) < d2))
             return false;
     }

     return true;
}

REAL   SquareDistPointPoly(const ElFifo<Pt2dr> & f,Pt2dr pt)
{
     if ( PointInPoly(f,pt))
        return 0;

     REAL d2 = 1e40;

     for (INT k=0; k<f.nb() ; k++)
         ElSetMin
         (
            d2,
            SegComp(f[k],f[k+1]).square_dist_seg(pt)
         );

     return d2;
}

void BoxPts(ElFifo<Pt2dr> & pts,Pt2dr & p0,Pt2dr & p1)
{
     p0 = pts[0];
     p1 = pts[0];

     for (INT k=0 ; k<pts.nb() ; k++)
     {
         p0.SetInf(pts[k]);
         p1.SetSup(pts[k]);
     } 
}


Box2dr BoxPts(const std::vector<Pt2dr> & pts)
{
     Box2dr aRes;
     aRes._p0 = pts[0];
     aRes._p1 = pts[0];

     for (INT k=0 ; k<int(pts.size()) ; k++)
     {
         aRes._p0.SetInf(pts[k]);
         aRes._p1.SetSup(pts[k]);
     } 
     return aRes;
}



bool HasInterOriente(const std::vector<Pt2dr> & f1,const std::vector<Pt2dr> & f2)
{
   for (int aK=0; aK<int(f1.size());  aK++)
      if (PointInPoly(f2,f1[aK]))
         return true;

   return false;
}

bool HasInter(const std::vector<Pt2dr> & f1,const std::vector<Pt2dr> & f2)
{
   return    HasInterOriente(f1,f2) 
          || HasInterOriente(f2,f1);
}

std::vector<Pt2dr> DilateHomotetik 
                   (
                       const std::vector<Pt2dr> & aVPts,
                       double aF,
                       const Pt2dr & aCentre
                   )
{
   std::vector<Pt2dr> aRes;

   for (int aK=0; aK<int(aVPts.size());  aK++)
   {
       aRes.push_back(aCentre+(aVPts[aK]-aCentre)*aF);
   }
   return aRes;
}

std::vector<Pt2dr> DilateHomotetikCdg
                   (
                       const std::vector<Pt2dr> & aVPts,
                       double aF
                   )
{
   return DilateHomotetik(aVPts,aF,barrycentre(aVPts));
}


/*********************************************************/
/*                                                       */
/*            Clipping and H                             */
/*                                                       */
/*********************************************************/

       // EventInterv  

EventInterv::EventInterv(REAL absc,bool entr) :
    _absc (absc),
    _entr (entr)
{
}


EventInterv::EventInterv(): 
     _absc (-1e12), 
     _entr(true) 
{}

bool EventInterv::entr() const {return _entr;}
REAL EventInterv::absc() const {return _absc;}

static bool CmpEI(const EventInterv & e1,const EventInterv e2)
{
    if (e1.absc() != e2.absc())
       return e1.absc() < e2.absc(); 
    else
       return  e1.entr() && (!e2.entr());
}


       // PileEvInterv  

void PileEvInterv::add_ev(EventInterv ev)
{
    _events.pushlast(ev);
}

void PileEvInterv::clear()
{
     _events.clear();
}

ElFilo<EventInterv> & PileEvInterv::events() 
{
    return _events;
}


void PileEvInterv::sort_ev()
{
    STDSORT
    (
         _events.tab(),
         _events.tab() + _events.nb(),
         CmpEI
    );
}


void IntervDisjoint::init(PileEvInterv & PEI)
{
    _intervs.clear();
    PEI.sort_ev();
    ElFilo<EventInterv> & fei = PEI.events();

    INT nb_entr = 0;
    REAL last_entr= 0.0;

    for (INT k=0; k<fei.nb() ; k++)
    {
         if (fei[k].entr())
         {
             if (nb_entr == 0)
                last_entr = fei[k].absc();
             nb_entr ++;
         }
         else
         {
             nb_entr --;
             if (nb_entr == 0)
                 _intervs.pushlast(Interval(last_entr,fei[k].absc()));
         }
    }
}

IntervDisjoint::IntervDisjoint(PileEvInterv & PEI)
{
     init(PEI);
}

IntervDisjoint::IntervDisjoint() {}

class ActionSeg
{
      public  :
  
          virtual void action_seg(const Seg2d &) = 0;

          IntervDisjoint _intervs;
          PileEvInterv   _events;
          virtual ~ActionSeg();
          ActionSeg() {};

      private :
};


ActionSeg::~ActionSeg() {}

void  ClipSeg
      (
           ActionSeg &             Act,
           ElFifo<Pt2dr> &   f,
           SegComp                 seg
      )
{
     f.set_circ(true);
     Act._events.clear();

     
     bool OrTrig = (surf_or_poly(f) >= 0);
     for (INT k=0; k<f.nb() ; k++)
     {
         Pt2dr p0 = seg.to_rep_loc(f[k]);
         Pt2dr p1 = seg.to_rep_loc(f[k+1]);

         if ((p0.y>0) != (p1.y>0))
         {
            bool entr = (p0.y>0);
            if (!OrTrig)
               entr = ! entr;
            REAL absc = p0.x-p0.y*((p1.x-p0.x)/(p1.y-p0.y))  ;  

            Act._events.add_ev(EventInterv(absc,entr));
         }
     }
     Act._intervs.init(Act._events);

     const ElFilo<Interval>  & intervs =   Act._intervs.intervs();


	 {
     for (INT k=0 ; k<intervs.nb() ; k++)
     {
         Pt2dr p0 (intervs[k]._v0,0.0);
         Pt2dr p1 (intervs[k]._v1,0.0);
         Act.action_seg
         (
              Seg2d
              (
                  seg.from_rep_loc(p0),
                  seg.from_rep_loc(p1)
              )
         );
     }
	 }
}

void  HPoly
      (
           ActionSeg &             Act,
           ElFifo<Pt2dr> &   f,
           Pt2dr                   dir,
           REAL                    esp
      )
{
     REAL omax = -1e50;
     REAL omin =  1e50;

     SegComp s0 (Pt2dr(0,0),dir);

     for (INT k=0; k<f.nb() ; k++)
     {
          REAL ord = s0.ordonnee(f[k]);
          ElSetMax(omax,ord);
          ElSetMin(omin,ord);
     }
     
     for (REAL ord = round_up(omin/esp) *esp; ord<omax; ord += esp)
     {
          Pt2dr p0 = s0.from_rep_loc(Pt2dr(0.0,ord));
          ClipSeg(Act,f,Seg2d(p0,p0+s0.tangente()));
     }
      
}

class HWind : public ActionSeg
{
      public  :
  
          virtual void action_seg(const Seg2d &);

          HWind(El_Window w,Line_St lst);

      private :

         El_Window _w;
         Line_St   _lst;
};

HWind::HWind(El_Window w,Line_St lst) :
   _w   (w),
   _lst (lst)
{
}

void HWind::action_seg(const Seg2d & seg)
{
    _w.draw_seg(seg.p0(),seg.p1(),_lst);
}

void El_Window::hach
    (
          ElFifo<Pt2dr> & poly,
          Pt2dr           dir,
          REAL            esp,
          Line_St         lst
    )
{
    HWind   HW(*this,lst);
    HPoly(HW,poly,dir,esp);
}

void El_Window::hach
    (
          std::vector<Pt2dr> & aVPoly,
          Pt2dr           dir,
          REAL            esp,
          Line_St         lst
    )
{
    ElFifo<Pt2dr> poly(aVPoly,true);
    HWind   HW(*this,lst);
    HPoly(HW,poly,dir,esp);
}


class cPushSeg  : public ActionSeg
{
    public :
       cPushSeg( std::vector<Seg2d> & aVSegs) :
          mVSegs (aVSegs)
       {
       }
       void action_seg(const Seg2d & aSeg)  {mVSegs.push_back(aSeg);}
    private :
       std::vector<Seg2d>  & mVSegs;
};

std::vector<Seg2d> SegComp::Clip(const std::vector<Pt2dr> &   aVP)
{
    ElFifo<Pt2dr> f((int) aVP.size());
    for (INT aK=0 ; aK<INT(aVP.size()) ; aK++)
       f.push_back(aVP[aK]);
    std::vector<Seg2d> aRes;
    cPushSeg aPS(aRes);
    ClipSeg(aPS,f,*this);
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
