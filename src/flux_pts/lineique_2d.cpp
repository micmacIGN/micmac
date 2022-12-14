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




/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/*******                                                                    ********/
/*******                                                                    ********/
/*******          LINES                                                     ********/
/*******                                                                    ********/
/*******                                                                    ********/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/


/***********************************************************************************/
/*                                                                                 */
/*    Trace_Digital_line                                                           */
/*                                                                                 */
/***********************************************************************************/



Trace_Digital_line::Trace_Digital_line(Pt2di p1,Pt2di p2,bool conx_8,bool include_p2)
{

     _u = p2 - p1;
     _u1 = best_4_approx(_u); // one, at least, is a four direction

      _u2 = second_freeman_approx(_u,conx_8,_u1);


     if ((_u1^_u2) < 0)  // whish that the couple _u1,_u2
        ElSwap(_u1,_u2);   // be trigonometrically oriented (== anti clokerwise)

     _p_cur = p1;
     _nb_pts = (conx_8 ? dist8(_u) : dist4(_u)) + (INT) include_p2;

      _delta_1 = _u ^ _u1;
      _delta_2 = _u ^ _u2;
      _delta   = 0;  // because _p_cur = p1 , so it is on the line p1,p2;
}




INT Trace_Digital_line::next_buf(INT *x,INT *y,INT sz_buf)
{
    INT nb;

    for (nb= 0; (nb<sz_buf) && (_nb_pts> 0); nb ++)
    {
        x[nb] = _p_cur.x;
        y[nb] = _p_cur.y;

        next_pt();
    }
    return nb;
}



template <class Type> void bitm_marq_line (Type ** im,INT tx,INT ty,Pt2di p1,Pt2di p2,INT val)
{
    static const int SzBuf = 100;
    INT tab_x[SzBuf];
    INT tab_y[SzBuf];

    Trace_Digital_line tdl(p1,p2,true,true);

    INT nb;
    while((nb = tdl.next_buf(tab_x,tab_y,SzBuf)))
    {
         INT x,y;

         for (int k=0 ; k<nb ; k++)
         {
              x = tab_x[k];
              y = tab_y[k];
              if ((x>=0) && (x<tx) && (y>=0) && (y<ty))
                 im[y][x] = val;
         }
    }
}

template  void bitm_marq_line (U_INT1 ** im,INT tx,INT ty,Pt2di p1,Pt2di p2,INT val);




template <class Type> void bitm_marq_line (Type ** im,INT tx,INT ty,Pt2di p1,Pt2di p2,INT val,REAL ray)
{
    static const int SzBuf = 100;
    INT tab_x[SzBuf];
    INT tab_y[SzBuf];

    Trace_Digital_line tdl(p1,p2,true,true);

    INT nb;
    INT IRay = round_up(ray-1);
    if (IRay <=0)
    {
        bitm_marq_line(im,tx,ty,p1,p2,val);
        return;
    }

    while((nb = tdl.next_buf(tab_x,tab_y,SzBuf)))
    {
         INT xc,yc;

         for (int k=0 ; k<nb ; k++)
         {
              xc = tab_x[k];
              yc = tab_y[k];
              

              for (INT kx = -IRay; kx <= IRay; kx++)
                  for (INT ky = -IRay; ky <= IRay; ky++)
                  {
                       if (ElSquare(kx)+ElSquare(ky) < ElSquare(ray))
                       {
                            INT x = xc + kx;
                            INT y = yc + ky;
                            if ((x>=0) && (x<tx) && (y>=0) && (y<ty))
                               im[y][x] = val;
                       }
                  }
         }
    }
}

template  void bitm_marq_line (U_INT1 ** im,INT tx,INT ty,Pt2di p1,Pt2di p2,INT val,REAL ray);










/***********************************************************************************/
/*                                                                                 */
/*    Compute_Dig_line                                                             */
/*                                                                                 */
/***********************************************************************************/


class Compute_Dig_line : public Std_Flux_Of_Points<INT>
{
    public :


         Compute_Dig_line
         (
             const  Arg_Flux_Pts_Comp &   arg,
             bool         conx_8,
             bool         closed,
             ElList<Pt2di>  l,
             bool           ForPoly
         );

    private :
         const Pack_Of_Pts * next();

         Trace_Digital_line  _tdl;
         ElList<Pt2di>    _l;
         Pt2di          _p1;
         Pt2di          _p2;
         bool _conx_8;
         bool  _closed;
};


Compute_Dig_line::Compute_Dig_line
(
    const  Arg_Flux_Pts_Comp &   arg,
    bool         conx_8,
    bool         closed,
    ElList<Pt2di>  l,
    bool           ForPoly 
)  :
          Std_Flux_Of_Points<INT>(2,arg.sz_buf())
{



    _p1       = l.pop();
    _p2       = l.pop();
    _l        = l;
    _closed   = closed;
    _conx_8   = conx_8;
    _tdl = Trace_Digital_line(_p1,_p2,_conx_8,l.empty()|| (ForPoly&&(_p1==_p2)));
}


const Pack_Of_Pts * Compute_Dig_line::next()
{
    while(1)
    {
         if (INT nb = _tdl.next_buf(_pack->_pts[0],_pack->_pts[1],sz_buf()))
         {
             _pack->set_nb(nb);
             return _pack;
         }

         if (_l.empty())
            return 0;
         _p1 = _p2;
         _p2 = _l.pop();
         _tdl = Trace_Digital_line(_p1,_p2,_conx_8,_l.empty()&&(!_closed));
    }
}



/***********************************************************************************/
/*                                                                                 */
/*    Dig_line_Not_Comp                                                            */
/*                                                                                 */
/***********************************************************************************/

class Dig_line_Not_Comp : public Flux_Pts_Not_Comp
{
     public :  

         Dig_line_Not_Comp(bool conx_8,bool closed,ElList<Pt2di> l,bool ForPoly);
         virtual  Flux_Pts_Computed * compute(const Arg_Flux_Pts_Comp &);

     private :
          bool         _conx_8;
          bool         _closed;
          ElList<Pt2di>  _l;
          bool           mForPoly;
};

Dig_line_Not_Comp::Dig_line_Not_Comp
(
   bool         conx_8,
   bool         closed,
   ElList<Pt2di>  l,
   bool           ForPoly
) :
  _conx_8 (conx_8),
  _closed (closed),
  _l      (l),
  mForPoly (ForPoly)
{
}


Flux_Pts_Computed * Dig_line_Not_Comp::compute(const Arg_Flux_Pts_Comp & arg)
{
     ASSERT_TJS_USER(_l.card()>=2,"poly-line with nb point < 2");
     if (_closed)
     {
        _l = _l + _l.last();
     }

     return new Compute_Dig_line(arg,_conx_8,_closed,_l,mForPoly);
}


Flux_Pts line_gen(ElList<Pt2di>  l,bool conx_8,bool closed,bool ForPoly)
{
    return new Dig_line_Not_Comp (conx_8,closed,l,ForPoly);
}


Flux_Pts line(Pt2di p1,Pt2di p2)
{
    return line_gen(NewLPt2di(p2)+p1,true,false,false);
}

Flux_Pts line_4c(Pt2di p1,Pt2di p2)
{
    return line_gen(NewLPt2di(p2)+p1,false,false,false);
}

Flux_Pts line(ElList<Pt2di>  l,bool closed)
{
    return line_gen(l,true,closed,false);
}

Flux_Pts line_4c(ElList<Pt2di>  l,bool closed)
{
    return line_gen(l,false,closed,false);
}


Flux_Pts line_for_poly(ElList<Pt2di>  l)
{
    return line_gen(l,true,true,true);
}       


/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/*******                                                                    ********/
/*******                                                                    ********/
/*******          MAPPING OF 2D-RECTANGLES BY DIGITAL LINES                 ********/
/*******                                                                    ********/
/*******                                                                    ********/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/


class Line_Map_Rect_Comp : public Std_Flux_Of_Points<INT>
{
     public :


        Line_Map_Rect_Comp
        (
            Pt2di  u,
            Pt2di  p0,
            Pt2di  p1,
            INT    aRabBuf =0
        );

        void Init(
            Pt2di  u,
            Pt2di  p0,
            Pt2di  p1
        );

        virtual ~Line_Map_Rect_Comp()
        {
            delete _pck_tr;
            delete _res;
        }

        virtual const Pack_Of_Pts * next();
     private :
        virtual bool is_line_map_rect() {return true;}
        virtual REAL average_dist() { return average_euclid_line_seed(_u);}


        bool   _horiz;
        Pt2di  _u;          // the direction of the line
        Pt2di  _p0;         // limit of rectangle
        Pt2di  _p1;         // limit of rectangle
        Pt2di  _trans;

        INT    * _x_ori;
        INT    * _y_ori;

        INT    * _x_tr;
        INT    * _y_tr;


        Std_Pack_Of_Pts<INT> * _pck_tr;
        Std_Pack_Of_Pts<INT> * _res;   // an intervale of _pck_tr

        INT    _i_beg;  // [_i_beg,_i_end[ interval of segment inside the box
        INT    _i_end;

        
        inline bool in_box(INT ind)
        {
              return 
                 (_trans + Pt2di(_x_ori[ind],_y_ori[ind])).in_box(_p0,_p1);
        }

};


const Pack_Of_Pts *  Line_Map_Rect_Comp::next()
{
    if ( (! (in_box(_i_beg))) && (! (in_box(_i_end-1))))
      return 0;

    while (! (in_box(_i_beg)))
         _i_beg++;

    while (in_box(_i_beg-1))
         _i_beg--;


    while (in_box(_i_end))
         _i_end ++;
    while (! (in_box(_i_end-1)))
         _i_end--;


    if (_horiz)
    {
       for (int i = _i_beg; i <_i_end; i++)
           _y_tr[i] = _y_ori[i] +_trans.y;
       _trans.y ++;
    }
    else
    {
       for (int i = _i_beg; i <_i_end; i++)
           _x_tr[i] = _x_ori[i] +_trans.x;
       _trans.x ++;
    }

    _res->interv(_pck_tr,_i_beg,_i_end);

    return _res;
}


static INT BufNeeded
       (  
         Pt2di  u,
         Pt2di  p0,
         Pt2di  p1
       )
{
   return  2+ ((ElAbs(u.x)>ElAbs(u.y)) ? ElAbs(p0.x-p1.x) : ElAbs(p0.y-p1.y));
}


Line_Map_Rect_Comp::Line_Map_Rect_Comp
(
         Pt2di  u,
         Pt2di  p0,
         Pt2di  p1,
         INT    RabU
) :
  // sz of buf : (if u is rather horizontal witdh else heigth) + 2 (for extremities)
  Std_Flux_Of_Points<INT> (2,ElMax(BufNeeded(Pt2di(1,0),p0,p1),BufNeeded(Pt2di(0,1),p0,p1)))
{
   _res = Std_Pack_Of_Pts<INT>::new_pck(2,0);
   _pck_tr  = 0;
   Init(u,p0,p1);
}


void Line_Map_Rect_Comp::Init ( Pt2di  u, Pt2di  p0, Pt2di  p1)
{
  INT aSzBLoc = BufNeeded(u,p0,p1);
  ELISE_ASSERT(Flux_Pts_Computed ::sz_buf()>=aSzBLoc,"Not Enough Buf in  Line_Map_Rect_Comp");

  _horiz                  =ElAbs(u.x)>ElAbs(u.y);
  _u                      =u;
  _p0                     =Inf(p0,p1);
  _p1                     =Sup(p0,p1);
  _trans                  =Pt2di(0,0);

    Trace_Digital_line  tdl (
                               Pt2di(0,0),
                               u*(1+aSzBLoc/dist8(u)),
                               true      ,
                               true
                           );
   _x_ori = _pack->_pts[0];
   _y_ori = _pack->_pts[1];
    tdl.next_buf(_x_ori+1,_y_ori+1,aSzBLoc-2);
   _pack->set_nb(_pack->pck_sz_buf());


   {
       Pt2di first (_x_ori[1],_y_ori[1]);
       Pt2di last  (_x_ori[aSzBLoc-2],_y_ori[aSzBLoc-2]);
       Pt2di pmax = Sup(first,last);
       Pt2di pmin = Inf(first,last);


       // if (_horiz)
       // set the upper left corner of  seg's box on bottom left corner of box [p1,p2]
       // set the botom right corner of seg's box on  down left of box [p1,p2]


       Pt2di tr = _horiz                                                         ?
                  (     corner_box_included(_p0,_p1,true,true)
                      - corner_box_included(pmin,pmax+Pt2di(1,1),true,false)
                  )                                                              :
                  (     corner_box_included(_p0,_p1,true,true)
                      - corner_box_included(pmin,pmax+Pt2di(1,1),false,true)
                  )                                                              ;
       for (int i =1 ; i <aSzBLoc-1; i++)
       {
           _x_ori[i] += tr.x;
           _y_ori[i] += tr.y;
       }
   }


    if (in_box(1))
    {
       _i_beg = 1;
    }
    else
    {
        if (in_box(aSzBLoc-2))
           _i_beg = aSzBLoc-2;
        else
           // artificial but will correctly leads to empty set
           _i_beg = 1;
    }
    _i_end = _i_beg +1;



   // put at extremities of pts original any points out of rectangle
   _x_ori[0] = _x_ori[aSzBLoc-1] = _p0.x-10;
   _y_ori[0] = _y_ori[aSzBLoc-1] = _p0.y-10;



   delete  _pck_tr;
   _pck_tr = SAFE_DYNC(Std_Pack_Of_Pts<INT> *,_pack->dup(_pack->pck_sz_buf()));
   _x_tr = _pck_tr->_pts[0];
   _y_tr = _pck_tr->_pts[1];


}




class Line_Map_Rect_Not_Comp : public Flux_Pts_Not_Comp
{
    public :


       Line_Map_Rect_Not_Comp(Pt2di u,Pt2di p0,Pt2di p1);

   private :
       Flux_Pts_Computed * compute (const  Arg_Flux_Pts_Comp & )
       {
            Flux_Pts_Computed * res =
                    new Line_Map_Rect_Comp(_u,_p0,_p1);
            return res;
       }

        Pt2di  _u;          // the direction of the line
        Pt2di  _p0;         // limit of rectangle
        Pt2di  _p1;         // limit of rectangle
          
};


Line_Map_Rect_Not_Comp::Line_Map_Rect_Not_Comp(Pt2di u,Pt2di p0,Pt2di p1) :
    _u  (u),
    _p0 (p0),
    _p1 (p1)
{
    ASSERT_TJS_USER
    (
         dist8(u) != 0,
         "null point in Line_Map_Rect_Not_Comp"
    );
}



Flux_Pts line_map_rect(Pt2di u,Pt2di p0,Pt2di p1)
{
    return new Line_Map_Rect_Not_Comp(u,p0,p1);
}


         /******************************************/
         /*                                        */
         /*     cLineMapRect                       */
         /*                                        */
         /******************************************/

cLineMapRect::~cLineMapRect()
{
   delete mPLMRP;
}

cLineMapRect::cLineMapRect(Pt2di aSzMax)  :
    mPLMRP (new Line_Map_Rect_Comp(Pt2di(1,1),Pt2di(0,0),aSzMax))
{
}

void cLineMapRect::Init(Pt2di u,Pt2di p0,Pt2di p1)
{
    mPLMRP->Init(u,p0,p1);
}




const cLineMapRect::tContPts * cLineMapRect::Next()
{
     const Pack_Of_Pts * pck = mPLMRP->next();

     if (pck==0)
     {
         return 0;
     }

     const Std_Pack_Of_Pts<INT> * anIPack = pck->int_cast ();
     INT * pX = anIPack->_pts[0];
     INT * pY = anIPack->_pts[1];
     INT aNb =  anIPack->nb();

     mCPts.clear();
     for (INT k=0; k<aNb ; k++)
          mCPts.push_back(Pt2di(pX[k],pY[k]));

     return & mCPts;
}



/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/*******                                                                    ********/
/*******                                                                    ********/
/*******          MAPPING OF 1D-RECTANGLES BY ``DIGITAL LINES''             ********/
/*******                                                                    ********/
/*******                                                                    ********/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/


           /******************************************************/
           /*                                                    */
           /*        Line_1d_Map_Rect_Comp                       */
           /*                                                    */
           /******************************************************/

class Line_1d_Map_Rect_Comp : public Std_Flux_Of_Points<INT>
{
     public :

        Line_1d_Map_Rect_Comp (INT  dx,INT  x0,INT  x1);

     private :
        virtual bool is_line_map_rect() {return true;}
        virtual REAL average_dist() { return 1.0;}

        virtual const Pack_Of_Pts * next();

        INT   _x0;         // limit of rectangle
        INT   _x1;         // limit of rectangle
        //INT   _dx;         // the direction of the line
        bool  _first;
};

Line_1d_Map_Rect_Comp::Line_1d_Map_Rect_Comp(INT  dx,INT  x0,INT  x1) :
          Std_Flux_Of_Points<INT>(1,ElAbs(x0-x1)),
          _x0    (ElMin(x0,x1)),
          _x1    (ElMax(x0,x1)),
          //_dx    (dx),
          _first (true)
{
    _pack->set_nb(_x1-_x0);
    for (int i=0,x = _x0; x<_x1; x++,i++)
        _pack->_pts[0][i] = x;
    if (dx < 0)
       _pack->auto_reverse();
}


const Pack_Of_Pts * Line_1d_Map_Rect_Comp::next()
{
   if (_first)
   {
      _first = false;
      return _pack;
   }
   else
      return 0;
}


           /******************************************************/
           /*                                                    */
           /*        Line_1d_Map_Rect_Not_Comp                   */
           /*                                                    */
           /******************************************************/

class Line_1d_Map_Rect_Not_Comp : public Flux_Pts_Not_Comp
{
    public :


       Line_1d_Map_Rect_Not_Comp(INT  dx,INT  x0,INT  x1);

   private :
       Flux_Pts_Computed * compute (const  Arg_Flux_Pts_Comp & )
       {
            Flux_Pts_Computed * res =
                    new Line_1d_Map_Rect_Comp(_dx,_x0,_x1);
            return res;
       }

        INT  _dx;          // the direction of the line
        INT  _x0;         // limit of rectangle
        INT  _x1;         // limit of rectangle
          
};


Line_1d_Map_Rect_Not_Comp::Line_1d_Map_Rect_Not_Comp(INT  dx,INT  x0,INT  x1) :
    _dx  (dx),
    _x0  (x0),
    _x1  (x1)
{
    ASSERT_TJS_USER
    (
         _dx != 0,
         "null point in Line_Map_Rect_Not_Comp"
    );
}



Flux_Pts line_map_rect(INT  dx,INT  x0,INT  x1)
{
    return new Line_1d_Map_Rect_Not_Comp(dx,x0,x1);
}

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/*******                                                                    ********/
/*******                                                                    ********/
/*******          CIRCLE                                                    ********/
/*******                                                                    ********/
/*******                                                                    ********/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/


/***********************************************************************************/
/*                                                                                 */
/*    Flux_By_Contour                                                              */
/*                                                                                 */
/***********************************************************************************/


class Flux_By_Contour : public Mcheck
{
    public :
         friend class Flx_Cont_Comp;

         Pt2di   p_first() { return _p_first;}
         Pt2di   p_end()   { return _p_end;}
  virtual ~Flux_By_Contour() {}

    protected: 


         static bool not_full_ang(REAL a0,REAL a1);

        Flux_By_Contour 
        (
            Pt2di p0,
            Pt2di q0,
            Pt2di p1,
            Pt2di q1,
            bool v8,
            bool not_full
        );
        void init();  // necessary to call in constor of inherited;
                       // because virtual call to inside


    private :

        INT k_next_pts(INT *x,INT *y,INT nb,bool & end);
        virtual bool inside(Pt2di) = 0;

        Pt2di  first_out(Pt2di in,Pt2di out,INT & k_prec);

        Config_Freeman_Or _freem;

        Pt2di   _p0;
        Pt2di   _q0;
        Pt2di   _p1;
        Pt2di   _q1;


        Pt2di   _p_first;
        Pt2di   _p_cur;
        Pt2di   _p_end;



        INT     _k_prec;
        bool    _not_full;
        INT     _nb_max; // use to control the special case of not_full
        bool    _first;
        bool    _init;

};

bool Flux_By_Contour::not_full_ang(REAL a0,REAL a1)
{
     REAL a = a1-a0;
     while (a>0) a -= 2* PI;
     while (a<=0) a += 2* PI;
     return a < PI;
}

Pt2di Flux_By_Contour::first_out(Pt2di in,Pt2di out,INT & k_prec)
{
     ASSERT_INTERNAL
     ( 
         inside(in) && (! inside(out)) ,
         "incoherence in Flux_By_Contour"
     );

     Trace_Digital_line tdl(in,out,false,true);

     k_prec = -1;
     Pt2di pprec;
     Pt2di p0; 
     while (tdl.next_buf(&p0.x,&p0.y,1))
     { 
         if (!inside(p0))
         {
            k_prec = _freem.num_pts(pprec-p0);
            return p0;
         }
         pprec = p0;
     }
     
     elise_internal_error("Flux_By_Contour",__FILE__,__LINE__);
     return Pt2di(0,0);
}

void Flux_By_Contour::init()
{
     _p_cur = first_out(_p0,_p1,_k_prec); 
     _p_first = _p_cur;
     {
        INT bid;
        _p_end = first_out(_q0,_q1,bid); 
     }
     _init = true;

     _nb_max =
                 (_not_full && (_p_end == _p_first))    ?
                 1                                      :
                 1000000000                             ;
}


Flux_By_Contour::Flux_By_Contour
(
        Pt2di p0,
        Pt2di q0,
        Pt2di p1,
        Pt2di q1,
        bool v8,
        bool not_full
) :
     _freem      (v8,true),
     _p0         (p0),
     _q0         (q0),
     _p1         (p1),
     _q1         (q1),
     _not_full   (not_full),
     _first      (true),
     _init       (false)
{
}


INT Flux_By_Contour::k_next_pts(INT *x,INT *y,INT nb,bool & end)
{
    end = false;
    ASSERT_INTERNAL(_init,"INIT probk in Flux_By_Contour");
    if (_first)
    {
         _first = false;
         while (inside(_p_cur+_freem.kth_p(_k_prec)))
           _k_prec = _freem.succ(_k_prec);
    }
    for (INT i=0; i<nb ; i++)
    {
        _nb_max--;
        x[i] = _p_cur.x;
        y[i] = _p_cur.y;
        // count  turn for special cas of 4 neigh (when all 4 neigh are outside)
        INT nb;
        for 
        (
            nb =0; 
            (nb<_freem.nb_pts()) && (!inside(_p_cur+_freem.kth_p(_k_prec)));
            nb++
        )
            _k_prec = _freem.succ(_k_prec);
        _k_prec =  _freem.prec(_k_prec); 
        _p_cur = _p_cur+_freem.kth_p(_k_prec);
        _k_prec =  _freem.sym(_k_prec);
        if ((_p_cur == _p_end) || (! _nb_max))
        {
           end = true;
           return (i+1);
        }
    }
    return nb;
}

/***********************************************************************************/
/*                                                                                 */
/*    Arc_Cercle_By_Contour                                                        */
/*                                                                                 */
/***********************************************************************************/


class Arc_Cercle_By_Contour : public Flux_By_Contour 
{
     public :
        Arc_Cercle_By_Contour (Pt2dr c,REAL r,REAL a0,REAL a1,bool v8);

     private :
        
        virtual bool inside(Pt2di p)
        {
            return ElSquare(p.x-_cx)+ElSquare(p.y-_cy)<_r2;
        }

        REAL _cx;
        REAL _cy;
        REAL _r2;
};

Arc_Cercle_By_Contour::Arc_Cercle_By_Contour
                 (Pt2dr c,REAL r,REAL a0,REAL a1,bool v8) :
          Flux_By_Contour
          (
              round_ni(c),
              round_ni(c),
              Pt2di(c+Pt2dr::FromPolar(r+3,a0)),
              Pt2di(c+Pt2dr::FromPolar(r+3,a1)),
              v8,
              Flux_By_Contour::not_full_ang(a0,a1)
          ),
          _cx (c.x),
          _cy (c.y),
          _r2  (ElSquare(r))
{
      init();
}

/***********************************************************************************/
/*                                                                                 */
/*    Arc_Ellipse_By_Contour                                                        */
/*                                                                                 */
/***********************************************************************************/


class Arc_Ellipse_By_Contour : public Flux_By_Contour 
{
     public :
        Arc_Ellipse_By_Contour (Pt2dr c,REAL A, REAL B, REAL teta,REAL a0,REAL a1,bool v8);

     private :
        
        virtual bool inside(Pt2di p)
        {
            REAL x = p.x-_cx;
            REAL y = p.y-_cy;

            return
                    (      ElSquare( _cos_t * x + _sin_t * y) / _A2
                       +   ElSquare(-_sin_t * x + _cos_t * y) / _B2
                    )  < 1.0;
        }

        REAL _cx;
        REAL _cy;
        REAL _A2;
        REAL _B2;
        REAL _cos_t;
        REAL _sin_t;

       static Pt2dr pol_ellipse(Pt2dr c,REAL exc,REAL teta,REAL rho,REAL alpha)
       {
              Pt2dr u( cos(teta),sin(teta));
              Pt2dr v(-sin(teta),cos(teta));

              v = v * exc;

              return c + ((u * cos(alpha)  + v * sin(alpha)) * rho);
       }
};

Arc_Ellipse_By_Contour::Arc_Ellipse_By_Contour 
(     
          Pt2dr c,     
          REAL A,      
          REAL B,      
          REAL teta,     
          REAL a0,     
          REAL a1,     
          bool v8     
)  :
          Flux_By_Contour
          (
              round_ni(c),
              round_ni(c),
              Pt2di(pol_ellipse(c,A/B,teta,3+ElMax(A,B),a0)),
              Pt2di(pol_ellipse(c,A/B,teta,3+ElMax(A,B),a1)),
              v8,
              Flux_By_Contour::not_full_ang(a0,a1)
          ),
          _cx (c.x),
          _cy (c.y),
          _A2  (ElSquare(A)),
          _B2  (ElSquare(B)),
          _cos_t (cos(teta)),
          _sin_t (sin(teta))
{
      init();
}



/***********************************************************************************/
/*                                                                                 */
/*    Flx_Cont_Comp                                                                */
/*                                                                                 */
/***********************************************************************************/

class Flx_Cont_Comp  : public Std_Flux_Of_Points<INT>
{
    public :


         Flx_Cont_Comp
         (
             const  Arg_Flux_Pts_Comp &   arg,
             Flux_By_Contour              *
         );
         virtual ~Flx_Cont_Comp() { delete _cont;}


    private :
         const Pack_Of_Pts * next();
         Flux_By_Contour *   _cont;
         bool                _end_cont;

};

Flx_Cont_Comp::Flx_Cont_Comp
(
     const  Arg_Flux_Pts_Comp &   arg,
     Flux_By_Contour              * cont
)        :
    Std_Flux_Of_Points<INT>(2,arg.sz_buf()),
    _cont (cont),
    _end_cont(false)
{
}

const Pack_Of_Pts * Flx_Cont_Comp::next()
{
      if (_end_cont) return 0;

      INT  nb = _cont->k_next_pts
                 (
                      _pack->_pts[0],
                      _pack->_pts[1],
                      sz_buf(),
                      _end_cont
                 );
      _pack->set_nb(nb);
      return _pack;
}

/***********************************************************************************/
/*                                                                                 */
/*    Arc_Cercle_Not_Comp                                                          */  
/*                                                                                 */
/***********************************************************************************/

class Arc_Cercle_Not_Comp : public Flux_Pts_Not_Comp
{
    public :

       Arc_Cercle_Not_Comp(Pt2dr c,REAL r,REAL a0,REAL a1,bool v8);

       void  extre(Pt2di & p1,Pt2di & p2)
       {
             Arc_Cercle_By_Contour ac (_cE,_r,_a0,_a1,_v8);
             p1 = ac.p_first();
             p2 = ac.p_end();
       }

   private :

      
       Flux_Pts_Computed * compute (const  Arg_Flux_Pts_Comp & arg)
       {
            Flux_Pts_Computed * res =
            new  Flx_Cont_Comp
                 (
                      arg,
                      new Arc_Cercle_By_Contour(_cE,_r,_a0,_a1,_v8)
                      
                  );
            return res;
       }

       Pt2dr  _cE;
       REAL   _r;
       REAL   _a0;
       REAL   _a1;
       bool   _v8; 
          
};

Arc_Cercle_Not_Comp::Arc_Cercle_Not_Comp
(
      Pt2dr c,
      REAL r,
      REAL a0,
      REAL a1,
      bool v8
)  :
       _cE (c),
       _r (r),
       _a0 (a0),
       _a1 (a1),
       _v8 (v8)
{
    ASSERT_TJS_USER(ElAbs(r)>=1.0,"Elise circles must have radius >= 1.0");
}



Flux_Pts circle(Pt2dr c,REAL r,bool v8)
{
     return new Arc_Cercle_Not_Comp(c,r,0,0,v8);
}


Flux_Pts arc_cir(Pt2dr c,REAL r,REAL a0,REAL a1,bool v8)
{
     return new Arc_Cercle_Not_Comp(c,r,a0,a1,v8);
}


Flux_Pts fr_sect_chord_ang(Pt2dr c,REAL r,REAL a0,REAL a1,bool v8,bool sect)
{
     Arc_Cercle_Not_Comp *ac  =  new Arc_Cercle_Not_Comp(c,r,a0,a1,v8);

     Pt2di p1,p2;
     ac->extre(p1,p2);

     LPt2di  l= (sect ? (NewLPt2di(p1)+Pt2di(c)+p2) : (NewLPt2di(p1) + p2));

     return 
             Flux_Pts(ac) 
         ||  (v8 ?  line(l) : line_4c(l));
}

Flux_Pts fr_sector_ang(Pt2dr c,REAL r,REAL a0,REAL a1,bool v8)
{
         return fr_sect_chord_ang(c,r,a0,a1,v8,true);
}

Flux_Pts fr_chord_ang(Pt2dr c,REAL r,REAL a0,REAL a1,bool v8)
{
         return fr_sect_chord_ang(c,r,a0,a1,v8,false);
}



/***********************************************************************************/
/*                                                                                 */
/*    Arc_Ellispe_Not_Comp                                                         */  
/*                                                                                 */
/***********************************************************************************/

class Arc_Ellispe_Not_Comp : public Flux_Pts_Not_Comp
{
    public :


       Arc_Ellispe_Not_Comp
       (     
          Pt2dr c,     
          REAL A,      
          REAL B,      
          REAL teta,     
          REAL a0,     
          REAL a1,     
          bool v8     
       );

       void  extre(Pt2di & p1,Pt2di & p2)
       {
             Arc_Ellipse_By_Contour ae (_cE,_AE,_BE,_teta,_a0,_a1,_v8);
             p1 = ae.p_first();
             p2 = ae.p_end();
       }


   private :
       Flux_Pts_Computed * compute (const  Arg_Flux_Pts_Comp & arg)
       {
            Flux_Pts_Computed * res =
            new  Flx_Cont_Comp
                 (
                      arg,
                      new Arc_Ellipse_By_Contour(_cE,_AE,_BE,_teta,_a0,_a1,_v8)
                      
                 );
            return res;
       }

      Pt2dr   _cE;
      REAL    _AE;
      REAL    _BE;
      REAL    _teta;
      REAL    _a0;
      REAL    _a1;
      bool    _v8;

};


Arc_Ellispe_Not_Comp::Arc_Ellispe_Not_Comp
(     
          Pt2dr c,     
          REAL A,      
          REAL B,      
          REAL teta,     
          REAL a0,     
          REAL a1,     
          bool v8     
)  :
    _cE       (c),
    _AE      (A),
    _BE       (B),
    _teta    (teta),
    _a0      (a0),
    _a1      (a1),
    _v8      (v8)
{
    ASSERT_TJS_USER(ElMin(ElAbs(A),ElAbs(B))>=1.0,"Elise elipse must have axes >= 1.0");
}



Flux_Pts ellipse(Pt2dr c,REAL A,REAL B,REAL teta,bool v8)
{
     return new Arc_Ellispe_Not_Comp(c,A,B,teta,0,0,v8);
}

Flux_Pts arc_ellipse(Pt2dr c,REAL A,REAL B,REAL teta,REAL a0,REAL a1,bool v8)
{
     return new Arc_Ellispe_Not_Comp(c,A,B,teta,a0,a1,v8);
}


Flux_Pts fr_sect_chord_ell(Pt2dr c,REAL A,REAL B,REAL teta,REAL a0,REAL a1,bool v8,bool sect)
{
     Arc_Ellispe_Not_Comp *ae  =  new Arc_Ellispe_Not_Comp(c,A,B,teta,a0,a1,v8);

     Pt2di p1,p2;
     ae->extre(p1,p2);

     LPt2di  l= (sect ? (NewLPt2di(p1)+Pt2di(c)+p2) : (NewLPt2di(p1) + p2));

     return Flux_Pts(ae) || (v8 ?  line(l) : line_4c(l));
}

Flux_Pts fr_sector_ell(Pt2dr c,REAL A,REAL B,REAL teta,REAL a0,REAL a1,bool v8)
{
          return fr_sect_chord_ell(c,A,B,teta,a0,a1,v8,true);
}

Flux_Pts fr_chord_ell(Pt2dr c,REAL A,REAL B,REAL teta,REAL a0,REAL a1,bool v8)
{
          return fr_sect_chord_ell(c,A,B,teta,a0,a1,v8,false);
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
