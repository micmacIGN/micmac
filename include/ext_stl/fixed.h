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



#ifndef _ELISE_EXT_STL_FIXED
#define _ELISE_EXT_STL_FIXED


template <const INT b> class ElFixed
{
     public :
         enum
         {
              b1 = b,  // Pour y acceder a partir du type par T::b1
              b2 = 2*(INT) b,
              Q  = 1 << (INT) b,
              QS2  = 1 << ((INT) b-1),
              Q2 = (1<<(INT) b2)
         };
};



template <const INT b>  class ElPFixed : public ElFixed<b>
{
     private :

        ElPFixed(INT x,INT y)   : _x(x), _y(y) {}
     public :

	// LLC : Low Level Create
	static ElPFixed<b> LLC(INT anX,INT anY){return ElPFixed<b>(anX,anY);}
        ElPFixed()   : _x(0), _y(0) {}
        INT  _x;
        INT  _y;

       ElPFixed (const Pt2dr & pt ) :
		_x ((INT)(pt.x*ElFixed<b>::Q)) ,
             _y ((INT)(pt.y*ElFixed<b>::Q))
       {}

       bool inside(const Pt2di & p0,const Pt2di & p1)
       {
            return    
                       ((p0.x<<b) <= _x)
                  &&   ((p0.y<<b) <= _y)
                  &&   ((p1.x<<b) >  _x)
                  &&   ((p1.y<<b) >  _y);
       }
                                                      
       Pt2di  Pt2diConv() const
       {
             return Pt2di
                    (
                         (_x+this->QS2)>>b,
                         (_y+this->QS2)>>b
                    );
       }

        REAL x() const {return _x/(REAL) this->Q;}
        REAL y() const {return _y/(REAL) this->Q;}

       Pt2dr  Pt2drConv() const {return Pt2dr(x(),y());}

#if (ELISE_unix || ELISE_MacOs || ELISE_MinGW)
        friend std::ostream & operator << (std::ostream & ofs,const ElPFixed  &p)
        {
               ofs << "[x " << p.x() << " ;y " << p.y() <<"]";
               return ofs;
        }  
#endif
		
        ElPFixed<b> operator - (const ElPFixed<b> & p2)
        {
                return ElPFixed (_x-p2._x, _y-p2._y);
        }
        ElPFixed<b> operator + (const ElPFixed<b> & p2)
        {
                return ElPFixed (_x+p2._x, _y+p2._y);
        }
        void operator += (const ElPFixed<b> & p2){ _x+= p2._x;_y+=p2._y;}

        bool  operator == (const ElPFixed<b> & p2) const 
        {
                 return (_x==p2._x) && (_y==p2._y);
        }
        ElPFixed<b> operator * (INT aScal) const {return  ElPFixed<b>(_x*aScal,_y*aScal);}
        ElPFixed<b> MulRat (INT  aF,INT aNBB) const 
        {
              return  ElPFixed<b>((_x*aF)>>aNBB,(_y*aF)>>aNBB);
        }



        ElPFixed<b> operator + (const Pt2di & p2)
        {
                return ElPFixed
                       (
                             _x+(p2.x<<b),
                             _y+(p2.y<<b)
                        );
        }

		 ElPFixed<b> operator * (const ElPFixed<b> & p2) 
	     {
             return  ElPFixed<b>
             (
                    (_x*p2._x-_y*p2._y) >> b,
                    (_x*p2._y+_y*p2._x) >> b
             );
	     }


        void operator += (const Pt2di & p2)
        {
             _x+= (p2.x<<b);
             _y+= (p2.y<<b);
        }

        void AddScalFixed(const ElPFixed<b> & p2, const INT & aScalFixed)
        {
             _x+= ((p2.x*this->aSF)<<this->b2);
             _y+= ((p2.y*this->aSF)<<this->b2);
        }

};
                         
/*
template <const INT b1,const INT b2>
         ElPFixed<b1> operator * (const ElPFixed<b1> & p1,const ElPFixed<b2> & p2)
{
      return ElPFixed<b1>
             (
                    (p1._x*p2._x-p1._y*p2._y) >> b2,
                    (p1._x*p2._y+p1._y*p2._x) >> b2
             );
}                                                                            
*/

// Version qui evite les 
// Error: Non-type template parameters are not allowed for function templates
// sur les vieux compilos.

/*
template <class T1>
         T1 operator * (const T1 & p1,const T1 & p2)
{
      return T1
             (
                    (p1._x*p2._x-p1._y*p2._y) >> T1::b1,
                    (p1._x*p2._y+p1._y*p2._x) >> T1::b1
             );
}                                                                            
*/


template <const INT b> class ElSegIter : public  ElFixed<b>
{
     public :
          ElSegIter(ElPFixed<b> p0,ElPFixed<b> p1,INT NB) :
               _x0        (p0._x),
               _y0        (p0._y),
               _dx        (p1._x-p0._x),
               _dy        (p1._y-p0._y),
               _dx_cur    (-_dx),
               _dy_cur    (-_dy),
               _bs        (round_ni(El_logDeux(NB))),
               _Qs        (1<<_bs),
               _k         (-1)
          {
          }

          bool next(ElPFixed<b> & p)
          {
               _k++;
               _dx_cur += _dx;
               _dy_cur += _dy;
               p._x = _x0 + (_dx_cur >> _bs);
               p._y = _y0 + (_dy_cur >> _bs);
               return _k <= _Qs;
          }
         
     private :

         INT   _x0;
         INT   _y0;
         INT   _dx;
         INT   _dy;
         INT   _dx_cur;
         INT   _dy_cur;
         INT   _bs;
         INT   _Qs;
         INT   _k;
};

#endif // _ELISE_EXT_STL_FIXED








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
