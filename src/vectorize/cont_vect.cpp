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
#include "ext_stl/pack_list.h"



class Cont_Explore_Cycle
{
      public :
         Cont_Explore_Cycle
         (
             Cont_Vect_Action  * act,
             ElFifo<Pt2di> &     pts, 
             Im2D_U_INT1         im
         );

         void explore_cycle(Pt2di p,INT k_prec);
         void explore_cycle();

      private :

         void next_kpt();

         Cont_Vect_Action  * _act;
         ElFifo<Pt2di> &     _pts;
         U_INT1 **           _data;
         INT                 _tx;
         INT                 _ty;
         Pt2di               _p;
         INT                 _k;
         INT                 _delta;
};

Cont_Explore_Cycle:: Cont_Explore_Cycle 
(
     Cont_Vect_Action  * act,
     ElFifo<Pt2di> &     pts, 
     Im2D_U_INT1         im
)  :
    _act (act),
    _pts (pts),
    _data (im.data()),
    _tx   (im.tx()),
    _ty   (im.ty())
{
}


void Cont_Explore_Cycle::next_kpt()
{
    _pts.pushlast(_p);
    _delta -= 3;
    _k = TAB_8_FREEM_SUCC_TRIG[TAB_8_FREEM_SYM[_k]];
    while(!( _data[_p.y][_p.x] & (1<< _k)))
    {
         _k =  TAB_8_FREEM_SUCC_TRIG[_k];
         _delta++;
    }
    _p += TAB_8_NEIGH[_k];
}

void Cont_Explore_Cycle::explore_cycle (Pt2di p,INT k)
{
    p += TAB_8_NEIGH[k];

   _pts.set_circ(true);
   Pt2di p0 = p;
   INT   k0 = k;

   _pts.clear();
   _p = p;
   _k = k;
   _delta = 0;

   next_kpt();
   while ((_p != p0) || (_k != k0))
      next_kpt();

   ELISE_ASSERT(ElAbs(_delta) == 8,"cycle dec");
   _act->action(_pts,_delta == 8);

   for (INT ip =0; ip<_pts.nb() ; ip++)
   {
       Pt2di p1 = _pts[ip];
       Pt2di p2 = _pts[ip+1];
       INT k =  freeman_code(p2-p1);
       _data[p1.y][p1.x] ^= (1<<k);
   }
}                                      

void Cont_Explore_Cycle::explore_cycle ()
{
     INT x,y,k;

     for (y=0 ; y<_ty ; y++)
         for (x=0 ; x<_tx ; x++)
             if (_data[y][x])
                for(k=0 ; k<8 ; k++)
                   if (_data[y][x] & (1<<k))
                      explore_cycle(Pt2di(x,y),k);
}


void explore_cyle
     ( 
           Cont_Vect_Action  * act,
           ElFifo<Pt2di> &     pts,
           Im2D_U_INT1         im     
     )
{
     Cont_Explore_Cycle CEP(act,pts,im);
     CEP. explore_cycle();
}

/********************************************************************/
/********************************************************************/
/********************************************************************/

static INT CPT = 0;



#include <map>

#define NB_PTS_LCONT   24
typedef  ElPackList<U_INT1,NB_PTS_LCONT>    ContVectLCodes;


class TyPt3dCmp
{

    public :
       bool operator ()  (const Pt3di& p0,const Pt3di& p1) const
       {
           if (p0.x < p1.x) return true;
           if (p0.x > p1.x) return false;

           if (p0.y < p1.y) return true;
           if (p0.y > p1.y) return false;

           if (p0.z < p1.z) return true;
           return false;
       }
};

TyPt3dCmp Pt3dCmp;

Pt3di IndBr_Cont_Vect(const  ElFifo<Pt2di> & pts,INT k,Pt2di p0)
{
    INT ind_freem = freeman_code(pts[k+1]-pts[k]);
    ELISE_ASSERT(ind_freem>=0,"IndBr_Cont_Vect");
    return Pt3di(pts[k]+p0,ind_freem);
}

class Vecto_Contour;

class Br_Cont_Vect
{
       public :

           ~Br_Cont_Vect();
           Br_Cont_Vect
           (
                Vecto_Contour &,
                ContVectLCodes *,
                const  ElFifo<Pt2di> &,
                INT k0,
                INT k1
           );

           // res = cont externe
           bool make_pts(ElFifo<Pt2di> &);
       
           void fusion_avant(Br_Cont_Vect *);
           void fusion_apres(Br_Cont_Vect *);

           void * operator new    (size_t sz);
           void  operator delete   (void * ptr) ;

           Pt3di Ip0() const {return _Ip0;}
           Pt3di Ip1() const {return _Ip1;}

           INT size() {return _codes.size();}

       private :

           ContVectLCodes  _codes;
           Pt3di           _Ip0;
           Pt3di           _Ip1;
           Vecto_Contour & _vc;

};

ElListAlloc<sizeof(Br_Cont_Vect)> Br_Cont_Vect_Lall;

void * Br_Cont_Vect::operator new    (size_t sz)
{
    // return Br_Cont_Vect_Lall.get();
    void * res =  Br_Cont_Vect_Lall.get();

    return res;
}

void Br_Cont_Vect::operator delete(void * adr)
{
    Br_Cont_Vect_Lall.put(adr);
}

bool Br_Cont_Vect::make_pts(ElFifo<Pt2di> & pts)
{
    pts.clear();
    Pt2di pcur  = Pt2di(_Ip0.x,_Ip0.y);

    INT surf = 0;

    INT cprec = _codes.back();
    INT turn = 0;

    for 
    (
         ContVectLCodes::iterator it = _codes.begin();
         it != _codes.end();
         it++
     )
     {
          pts.pushlast(pcur);
          INT cod = *it;
          Pt2di pnext = pcur + TAB_8_NEIGH[cod];

          INT delta = *it - cprec;
          if (delta >4) delta -= 8;
          if (delta <=-4) delta += 8;
          turn += delta;

          surf += pcur ^ pnext;

          pcur = pnext;
          cprec = cod;
     }

     ELISE_ASSERT(ElAbs(turn)==8,"Check Sum In cycle dec");

     return turn >0;
}


class Vecto_Contour : public Simple_OPBuf1<INT,U_INT1>,
                      public Cont_Vect_Action
{
     public :
 
       enum 
       {
           RAB_PTS = 4,
	   NbPackY = 200
       };

       Vecto_Contour(Cont_Vect_Action* act,bool cx8);
       ~Vecto_Contour()
       {
          Br_Cont_Vect_Lall.purge();
       }


       void unindex(Br_Cont_Vect * br)
       {
           _DicBrIp0.erase(br->Ip0());
           _DicBrIp1.erase(br->Ip1());
       }

       void index(Br_Cont_Vect * br)
       {
           _DicBrIp0[br->Ip0()] = br;
           _DicBrIp1[br->Ip1()] = br;
       }

       Pt2di dec() {return _dec;}
       void show_dict();

 
     private :
 
       Fonc_Num adapt_box(Fonc_Num f,Box2di b)
       {
            return clip_def(f>0,0,b._p0,b._p1);
       }
       void  calc_buf(INT ** output,U_INT1 *** input);        
       Simple_OPBuf1<INT,U_INT1> * dup_comp();
       virtual void action
               (
                    const ElFifo<Pt2di> &,
                    bool                ext
               ) ;        

       bool inside_dom(Pt2di p)
       {
           return p.in_box(_p0_dom,_p1_dom);
       }

       void add_brins(const ElFifo<Pt2di> & pts,INT k0,INT k1,bool circ,bool ext);
       void use_brin_finish(const ElFifo<Pt2di> & pts,bool ext);


       Cont_Vect_Action *     _act;
       bool                   _cx8;
       Im2D_U_INT1             _im;
       ElFifo<Pt2di>  _buf_pts_excy;
       ElFifo<Pt2di>    _buf_pts_br;
       Pt2di                _p0_dom;
       Pt2di                _sz_dom;
       Pt2di                _p1_dom;
       Pt2di                _dec;

       typedef ElSTDNS map<Pt3di,Br_Cont_Vect *,TyPt3dCmp> TyMapPBr;
       TyMapPBr _DicBrIp0;
       TyMapPBr _DicBrIp1;
       ContVectLCodes  _BufLcodes;



       Br_Cont_Vect * FindEl (TyMapPBr & Dic,const Pt3di & p)
       {
            TyMapPBr::iterator it = Dic.find(p);
            return (it==Dic.end()) ? 0 : it->second;
       }


        Vecto_Contour(const Vecto_Contour & );

        void show_dict(const char *,TyMapPBr &);
};


void Vecto_Contour::show_dict(const char * Nm,Vecto_Contour::TyMapPBr & Dico)
{
    cout << "  #####-- " << Nm << " --####\n";
    for (TyMapPBr::iterator it = Dico.begin(); it!= Dico.end() ; it++)
    {
        cout << "    " << Nm << "[" << it->first << "] " << it->second << "\n";
    }
}

void Vecto_Contour::show_dict()
{
    show_dict("DP0",_DicBrIp0);
    show_dict("DP1",_DicBrIp1);
}



Br_Cont_Vect::~Br_Cont_Vect()
{
   _vc.unindex(this);
}


Br_Cont_Vect::Br_Cont_Vect
(
    Vecto_Contour & vc,
    ContVectLCodes * res,
    const  ElFifo<Pt2di> & pts,
    INT k0,
    INT k1
) :
  _codes (res),      
  _Ip0   (IndBr_Cont_Vect(pts,k0,vc.dec())),
  _Ip1   (IndBr_Cont_Vect(pts,k1,vc.dec())),
  _vc    (vc)
{
   _vc.index(this);
   for (INT k= k0; k<k1 ; k++)
       _codes.push_back(freeman_code(pts[k+1]-pts[k]));
}


void Br_Cont_Vect::fusion_avant(Br_Cont_Vect * br2)
{
    _vc.unindex(this);
    _Ip0 = br2->_Ip0;

    while (!  br2->_codes.empty())
    {
        _codes.push_front( br2->_codes.pop_back());
    }

    delete br2;
    _vc.index(this);
}


void Br_Cont_Vect::fusion_apres(Br_Cont_Vect * br2)
{
    _vc.unindex(this);
    _Ip1 = br2->_Ip1;

    while (!  br2->_codes.empty())
    {
        _codes.push_back( br2->_codes.pop_front());
    }

    delete br2;
    _vc.index(this);
}





void Vecto_Contour::use_brin_finish(const ElFifo<Pt2di> & pts,bool ext)
{

    _act->action(pts,ext);
}


void Vecto_Contour::add_brins
     (
         const ElFifo<Pt2di> & pts,
         INT k0,
         INT k1,
         bool circ,
         bool ext
     )
{
    if (circ)
    {
        _buf_pts_br.clear();
        for(INT k=0; k<pts.nb() ; k++)
           _buf_pts_br.pushlast(pts[k]+_dec);
        use_brin_finish(_buf_pts_br,ext);
        return;
    }


    Br_Cont_Vect * br = new Br_Cont_Vect(*this,&_BufLcodes,pts,k0,k1);
    Br_Cont_Vect * br_prec =  FindEl(_DicBrIp1,br->Ip0());
    Br_Cont_Vect * br_next =  FindEl(_DicBrIp0,br->Ip1());


    if ((br_prec==0) && (br_next==0))
    {
        return;
    }


    if ((br_prec!=0) && (br_next==0))
    {
        br_prec->fusion_apres(br);
        return;
    }


    if ((br_prec==0) && (br_next!=0))
    {
        br_next->fusion_avant(br);
        return;
/*
        unindex(br_next);
        br_next->fusion_avant(br);
        index(br_next);
        delete br;
*/
    }

    if (br_prec == br_next)
    {
        br_prec->fusion_apres(br);
        bool ext = br_prec->make_pts(_buf_pts_br);
        use_brin_finish(_buf_pts_br,ext);

        delete br_prec;
        return;
    }

    

    if (br_prec->size() > br_next->size())
    {
        br_prec->fusion_apres(br);
        br_prec->fusion_apres(br_next);
    }
    else
    {
        br_next->fusion_avant(br);
        br_next->fusion_avant(br_prec);
    }

}





void Vecto_Contour::action(const ElFifo<Pt2di> & pts,bool ext)
{
     INT k0=0;

     while ((k0<pts.nb()) && (! inside_dom(pts[k0])))
           k0++;
     // brin completement en dehors de la zone
     if (k0 == pts.nb())
        return;

     INT k1 = k0+1;
     while ((k1<pts.nb()+k0) && ( inside_dom(pts[k1])))
           k1++;

  
     // brin circulaire
     if (k1 == pts.nb()+k0)
     {
	 add_brins(pts,k0,k1,true,ext);
        return;
     }

     while ( inside_dom(pts[k0-1]))
           k0--;

     INT k00 = k0;

     while (k0 != k00+pts.nb())
     {
         add_brins(pts,k0,k1,false,false); 
         k0 = k1 +1;
         while (! inside_dom(pts[k0]))
               k0++;
         k1 = k0+1;
         while ( inside_dom(pts[k1]))
               k1++;
     }
}



Simple_OPBuf1<INT,U_INT1> * Vecto_Contour::dup_comp()
{
      Vecto_Contour * res = new Vecto_Contour(_act,_cx8);

      res->_im =  Im2D_U_INT1(SzXBuf(),SzYBuf ());

      res->_p0_dom = Pt2di(RAB_PTS,RAB_PTS);
      res->_sz_dom = Pt2di(tx(), nb_pack_y ());
      res->_p1_dom =  res->_p0_dom+res->_sz_dom;

      return res;
}

Vecto_Contour::Vecto_Contour
(
       Cont_Vect_Action* act,
       bool cx8
)  :
   _act       (act),
   _cx8       (cx8),
   _im        (1,1),
   _DicBrIp0  (Pt3dCmp),
   _DicBrIp1  (Pt3dCmp)
{
}





void Vecto_Contour::calc_buf (INT ** out, U_INT1 *** input) 
{
      MEM_RAZ(out[0]+x0(),x1()-x0());  
      if (! first_line_in_pack())    
         return;

      CPT++;
      _dec =  Pt2di(x0Buf(),y0Buf()+ycur());


      for (int y = 0 ; y < _im.ty() ; y++)
          convert
          (
              _im.data()[y],
              input[0][y+y0Buf()]+x0Buf(),
              SzXBuf()
          );          


      ELISE_COPY(_im.border(1),0,_im.out());

      if (_cx8)
         ELISE_COPY(_im.all_pts(), flag_front8(_im.in(0)),_im.out());
      else
         ELISE_COPY(_im.all_pts(), flag_front4(_im.in(0)),_im.out());

      explore_cyle(this,_buf_pts_excy,_im);

}



Fonc_Num  cont_vect
          (
               Fonc_Num f,
               Cont_Vect_Action * act,
               bool cx8
          )
{
     return create_op_buf_simple_tpl
            (
                new Vecto_Contour (act,cx8),
                f,
                1,
                Box2di(Vecto_Contour::RAB_PTS),
				Vecto_Contour::NbPackY
            );
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
