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


#define BUG 0

#include "StdAfx.h"

#if (ELISE_windows)&&(!ELISE_MinGW)
    // disable "potential divide by 0" warning in method void ElMatrix::SetColSchmidtOrthog(INT NbIter)
    #pragma warning( disable : 4723 )
#endif

static int cos(const int &)
{
       ELISE_ASSERT (false,"::cos(const int&)");
       return 0;
}
static int sin(const int &)
{
       ELISE_ASSERT (false,"::sin(const int&)");
       return 0;
}
static int sqrt(const int &)
{
       ELISE_ASSERT (false,"::sqrt(const int&)");
       return 0;
}


bool AcceptFalseRot = false;


/*************************************************************/
/*                                                           */
/*     Construction-Destruction etc ..                       */
/*                                                           */
/*************************************************************/

template <class Type>
         void ElMatrix<Type>::ResizeInside(INT TX,INT TY)
{
    if ((TX>mTMx) ||  (TY>mTMy))
    {
         set_to_size(ElMax(TX,mTMx),ElMax(TY,mTMy));
    }

    _tx = TX;
    _ty = TY;
}


template <class Type>
         void ElMatrix<Type>::init(INT TX,INT TY)
{
   mTMx = _tx = TX;
   mTMy = _ty = TY;
   _data = STD_NEW_TAB_USER(_ty,Type *);
   for (int y=0; y<_ty; y++)
        _data[y] = STD_NEW_TAB_USER(_tx,Type);
}

template <class Type>
         bool ElMatrix<Type>::same_size(const ElMatrix<Type> & m2) const
{
    return (_tx==m2._tx)  && (_ty==m2._ty) ;
}


template <class Type>
         void ElMatrix<Type>::dup_data(const ElMatrix<Type> & m2)
{
    ELISE_ASSERT(same_size(m2),"Diff size in ElMatrix::dup_data");
    for (int y=0; y<_ty; y++)
        for (INT x=0; x<_tx; x++)
            _data[y][x] = m2._data[y][x];
}

template <class Type>
         ElMatrix<Type>::ElMatrix(INT tx,INT ty,Type v)
{
    init(tx,ty);
    for (int y=0; y<_ty; y++)
        for (INT x=0; x<_tx; x++)
            _data[y][x] = v;
}

template <class Type>
         ElMatrix<Type>::ElMatrix(INT tx,bool init_id)
{

    init(tx,tx);
    for (int y=0; y<_ty; y++)
        for (INT x=0; x<_tx; x++)
        {
            if (init_id && (x==y))
               _data[y][x] = Type(1);
            else
               _data[y][x] = Type(0);
        }
}

template <class Type> ElMatrix<Type>::ElMatrix(const ElMatrix<Type> & m2)
{
    init(m2._tx,m2._ty);
    dup_data(m2);
}


template <class Type> void ElMatrix<Type>::set_to_size(INT TX,INT TY)
{
    if ((_tx != TX) || (_ty != TY))
    {
       un_init();
       init(TX,TY);
    }
}
template <class Type>
         void ElMatrix<Type>::set_to_size(const ElMatrix<Type> & m2)
{
     set_to_size(m2._tx,m2._ty);
}

template <class Type> ElMatrix<Type> &
                      ElMatrix<Type>::operator = (const ElMatrix<Type> & m2)
{
    if (this == &m2)
       return *this;

    set_to_size(m2);
    dup_data(m2);
    return *this;
}


template <class Type>  void ElMatrix<Type>::un_init()
{
    for (int y=0; y<mTMy; y++)
         STD_DELETE_TAB_USER(_data[y]);
    STD_DELETE_TAB_USER(_data);
    mTMx=0;
    mTMy=0;
}

template <class Type>  ElMatrix<Type>::~ElMatrix()
{
    un_init();
}

template <class Type>
         void ElMatrix<Type>::SetLine(INT NL,const Type *vals)
{
    ELISE_ASSERT((NL>=0)&&(NL<_ty),"Bad Line in ElMatrix<Type>::SetLine");
    Type * l = _data[NL];
    for (int x =0; x<_tx;x++)
        l[x] = vals[x];
}

template <class Type>
         void ElMatrix<Type>::GetLine(INT NL,Type *vals) const
{
    ELISE_ASSERT((NL>=0)&&(NL<_ty),"Bad Line in ElMatrix<Type>::SetLine");
    const Type * l = _data[NL];
    for (int x =0; x<_tx;x++)
         vals[x] = l[x];
}





template <class Type>  Type ElMatrix<Type>::ProdCC
                            (const ElMatrix<Type> & m2,INT x1,INT x2) const
{
    ELISE_ASSERT(_ty == m2._ty,"Diff size in ElMatrix::ProdCC");

    Type res =0;
    for (INT y=0; y<_ty ;y++)
        res = res + _data[y][x1] * m2._data[y][x2];
    return res;
}

template <class Type>  Type ElMatrix<Type>::ProdLC
                            (const ElMatrix<Type> & m2,INT y1,INT x2) const
{
    ELISE_ASSERT(_tx == m2._ty,"Diff size in ElMatrix::ProdLC");

    Type res =0;
    for (INT k=0;k<_tx; k++)
        res =   res +   _data[y1][k] * m2._data[k][x2];
    return res;
}

template <class Type>  Type ElMatrix<Type>::ProdLL
                            (const ElMatrix<Type> & m2,INT y1,INT y2) const
{
    ELISE_ASSERT(_tx == m2._tx,"Diff size in ElMatrix::ProdCC");

    Type res =0;
    for (INT x=0; x<_tx ;x++)
        res = res + _data[y1][x] * m2._data[y2][x];
    return res;
}



template <class Type> ElMatrix<Fonc_Num> ToMatForm(const  ElMatrix<Type> & aMat)
{
    ElMatrix<Fonc_Num> aRes(aMat.tx(),aMat.ty());

    for (INT x=0; x<aMat.tx(); x++)
        for (INT y=0; y<aMat.ty(); y++)
             aRes(x,y) = aMat(x,y);
    return aRes;
}
template  ElMatrix<Fonc_Num> ToMatForm(const  ElMatrix<REAL> &);
template  ElMatrix<Fonc_Num> ToMatForm(const  ElMatrix<INT> &);


/*************************************************************/
/*                                                           */
/*             Operations Matricielles                       */
/*                                                           */
/*************************************************************/

template <class Type>
          Type  ** ElMatrix<Type>::data()
{
    return _data;
}





template <class Type>  ElMatrix<Type>   gaussj(const ElMatrix<Type> & m)
{
    ELISE_ASSERT(m.tx()==m.ty(),"Non Square Matrice in Gaussj");
    ElMatrix<Type> res(m);
    gaussj(res.data(),res.tx());
    return res;
}

template <class Type>  void   self_gaussj(ElMatrix<Type> & m)
{
    ELISE_ASSERT(m.tx()==m.ty(),"Non Square Matrice in Gaussj");
    gaussj(m.data(),m.tx());
}


template <class Type>  bool   self_gaussj_svp(ElMatrix<Type> & m)
{
    ELISE_ASSERT(m.tx()==m.ty(),"Non Square Matrice in Gaussj");
    return gaussj_svp(m.data(),m.tx());
}




template <class Type>
         void ElMatrix<Type>::verif_addr_diff(const ElMatrix<Type> & m1)
{
    ELISE_ASSERT(this != &m1,"Addresses should be != in low level Matrix manip");
}

template <class Type>
         void ElMatrix<Type>::verif_addr_diff
         (
              const ElMatrix<Type> & m1,
              const ElMatrix<Type> & m2
         )
{
    verif_addr_diff(m1);
    verif_addr_diff(m2);
}


template <class Type>  void  ElMatrix<Type>::mul
                             (
                                  const ElMatrix<Type> & m1,
                                  const ElMatrix<Type> & m2
                             )
{

    verif_addr_diff(m1,m2);
    ELISE_ASSERT
    (
        m1._tx==m2._ty,
        "Incoherent size in ElMatrix mult"
    );

    set_to_size(m2._tx,m1._ty);

    for (INT x =0; x<_tx; x++)
        for (INT y =0; y<_ty; y++)
              _data[y][x] = m1.ProdLC(m2,y,x);
}


REAL EcartInv(const ElMatrix<REAL>& m1,const ElMatrix<REAL>& m2)
{
    ELISE_ASSERT ( m1.tx()==m2.ty(), "Incoherent size in EcartInv");

    REAL res = 0.0;

    for (INT x =0; x<m2.tx(); x++)
        for (INT y =0; y<m1.ty(); y++)
              res += ElSquare(m1.ProdLC(m2,y,x)-(x==y));
    return res;
}






template <class Type>
          ElMatrix<Type>   ElMatrix<Type>::operator *
                              (const ElMatrix<Type> & m2) const
{
    ElMatrix<Type> res (m2._tx,_ty);
    res.mul(*this,m2);
    return res;
}


template <class Type>  void  ElMatrix<Type>::mul
                             (const ElMatrix<Type> & m, const Type & v)
{

    // verif_addr_diff(m);
    set_to_size(m);

    for (INT x =0; x<_tx; x++)
        for (INT y =0; y<_ty; y++)
              _data[y][x] = m._data[y][x]*v;
}


template <class Type>
          ElMatrix<Type>   ElMatrix<Type>::operator *(const Type & v) const
{
    ElMatrix<Type> res (_tx,_ty);
    res.mul(*this,v);
    return res;
}

template <class Type>  void   ElMatrix<Type>::operator *=(const Type & v)
{
   mul(*this,v);
}


template <class Type> ElMatrix<Type>
                      operator *(const Type & v,const ElMatrix<Type>& m)
{
    ElMatrix<Type> res (m.tx(),m.ty());
    res.mul(m,v);
    return res;
}


template <class Type>  void  ElMatrix<Type>::add
                             (
                                  const ElMatrix<Type> & m1,
                                  const ElMatrix<Type> & m2
                             )
{

    ELISE_ASSERT
    (
        m1.same_size(m2),
        "Incoherent size in ElMatrix mult"
    );
    set_to_size(m1);

    for (INT x =0; x<_tx; x++)
        for (INT y =0; y<_ty; y++)
              _data[y][x] = m1._data[y][x]+m2._data[y][x];
}

template <class Type>  void  ElMatrix<Type>::operator +=
                              (const ElMatrix<Type> & m1)
{
     this->add(*this,m1);
}


template <class Type>
          ElMatrix<Type>   ElMatrix<Type>::operator +
                              (const ElMatrix<Type> & m2) const
{
    ElMatrix<Type> res (_tx,_ty);
    res.add(*this,m2);
    return res;
}


template <class Type>  void  ElMatrix<Type>::sub
                             (
                                  const ElMatrix<Type> & m1,
                                  const ElMatrix<Type> & m2
                             )
{

    ELISE_ASSERT
    (
        m1.same_size(m2),
        "Incoherent size in ElMatrix mult"
    );
    set_to_size(m1);

    for (INT x =0; x<_tx; x++)
        for (INT y =0; y<_ty; y++)
              _data[y][x] = m1._data[y][x]-m2._data[y][x];
}


template <class Type>
          ElMatrix<Type>   ElMatrix<Type>::operator -
                              (const ElMatrix<Type> & m2) const
{
    ElMatrix<Type> res (_tx,_ty);
    res.sub(*this,m2);
    return res;
}

template <class Type> Pt2d<Type> mul32
                      (const ElMatrix<Type> & M,const Pt3d<Type> & p)
{
    ELISE_ASSERT(M.tx()==3&&M.ty()==2,"Wrong size in  ElMatrix * Pt3d");
    return Pt2d<Type>
           (
               M(0,0)*p.x + M(1,0)*p.y +  M(2,0)*p.z ,
               M(0,1)*p.x + M(1,1)*p.y +  M(2,1)*p.z
           );
}



template <class Type> Pt2d<Type> operator *
         (const ElMatrix<Type> & M,const Pt2d<Type> &p)
{
    ELISE_ASSERT(M.tx()==2&&M.ty()==2,"Wrong size in  ElMatrix<Type> * Pt2d");
    return Pt2d<Type>
           (
               M(0,0)*p.x + M(1,0)*p.y,
               M(0,1)*p.x + M(1,1)*p.y
           );
}


template <class Type> Pt3d<Type> operator *
         (const ElMatrix<Type> & M,const Pt3d<Type> & p)
{
    ELISE_ASSERT(M.tx()==3&&M.ty()==3,"Wrong size in  ElMatrix * Pt3d");
    return Pt3d<Type>
           (
               M(0,0)*p.x + M(1,0)*p.y +  M(2,0)*p.z ,
               M(0,1)*p.x + M(1,1)*p.y +  M(2,1)*p.z ,
               M(0,2)*p.x + M(1,2)*p.y +  M(2,2)*p.z
           );
}

template <class Type> void ElMatrix<Type>::GetCol(INT col,Pt3d<Type> &  p) const
{
    ELISE_ASSERT(ty()==3,"Wrong size in  ElMatrix.GetCol(Pt3d)");
    p.x = (*this)(col,0);
    p.y = (*this)(col,1);
    p.z = (*this)(col,2);
}
template <class Type> void ElMatrix<Type>::GetCol(INT col,Pt2d<Type> &  p) const
{
    ELISE_ASSERT(ty()==2,"Wrong size in  ElMatrix.GetCol(Pt3d)");
    p.x = (*this)(col,0);
    p.y = (*this)(col,1);
}


template <class Type> void ElMatrix<Type>::GetLig(INT lig,Pt3d<Type> &  p) const
{
    ELISE_ASSERT(tx()==3,"Wrong size in  ElMatrix.GetCol(Pt3d)");
    p.x = (*this)(0,lig);
    p.y = (*this)(1,lig);
    p.z = (*this)(2,lig);
}
template <class Type> void ElMatrix<Type>::GetLig(INT lig,Pt2d<Type> &  p) const
{
    ELISE_ASSERT(tx()==2,"Wrong size in  ElMatrix.GetCol(Pt3d)");
    p.x = (*this)(0,lig);
    p.y = (*this)(1,lig);
}




template <class Type> void SetCol (ElMatrix<Type> & M,INT col,Pt3d<Type>  p)
{
    ELISE_ASSERT(M.ty()==3,"Wrong size in  ElMatrix.SetCol(Pt3d)");
    M(col,0) = p.x;
    M(col,1) = p.y;
    M(col,2) = p.z;
}

template <class Type> void SetLig (ElMatrix<Type> & M,INT lig,Pt3d<Type>  p)
{
    ELISE_ASSERT(M.tx()==3,"Wrong size in  ElMatrix.SetCol(Pt3d)");
    M(0,lig) = p.x;
    M(1,lig) = p.y;
    M(2,lig) = p.z;
}


template <class Type> void SetCol (ElMatrix<Type> & M,INT col,Pt2d<Type>  p)
{
    ELISE_ASSERT(M.ty()==2,"Wrong size in  ElMatrix.SetCol(Pt3d)");
    M(col,0) = p.x;
    M(col,1) = p.y;
}

template <class Type> void SetLig (ElMatrix<Type> & M,INT lig,Pt2d<Type>  p)
{
    ELISE_ASSERT(M.tx()==2,"Wrong size in  ElMatrix.SetCol(Pt3d)");
    M(0,lig) = p.x;
    M(1,lig) = p.y;
}






template <class Type>
         ElMatrix<Type> MatFromCol
            (Pt3d<Type> P0,Pt3d<Type> P1,Pt3d<Type> P2)
{
     ElMatrix<Type> Res(3,3);
     SetCol(Res,0,P0);
     SetCol(Res,1,P1);
     SetCol(Res,2,P2);

     return Res;
}

template <class Type>
         ElMatrix<Type> MatFromCol (Pt2d<Type> P0,Pt2d<Type> P1)
{
     ElMatrix<Type> Res(2,2);
     SetCol(Res,0,P0);
     SetCol(Res,1,P1);

     return Res;
}






ElMatrix<REAL> MatFromImageBase
               (
                     Pt3d<REAL> C0   ,Pt3d<REAL> C1   ,Pt3d<REAL> C2,
                     Pt3d<REAL> ImC0 ,Pt3d<REAL> ImC1 ,Pt3d<REAL> ImC2
               )
{
    return MatFromCol(ImC0,ImC1,ImC2) * gaussj(MatFromCol(C0,C1,C2));
}

template <class Type>  void  ElMatrix<Type>::transpose
                             (
                                const ElMatrix<Type> & m1
                             )
{
    verif_addr_diff(m1);
    set_to_size(m1._ty,m1._tx);

    for (INT x =0; x<_tx; x++)
        for (INT y =0; y<_ty; y++)
              _data[y][x] = m1._data[x][y];
}


void InspectPbCD(ElMatrix<REAL> aM)
{
  aM.SymetriseParleBas();
  ElMatrix<REAL> aValP(1,1),aVecP(1,1);

  std::vector<int>  aInd = jacobi_diag(aM,aValP,aVecP);
  for (int aK= 0 ; aK< int(aInd.size()) ; aK++)
  {
     std::cout << "VALp= " << aValP(aK,aK) << "\n";
  }
  for (int aX= 0 ; aX< int(aInd.size()) ; aX++)
  {
     for (int aY= 0 ; aY< int(aInd.size()) ; aY++)
     {
         printf("%lf ",aVecP(aX,aY));
     }
     printf("\n");
  }

}





template <class Type>  void  ElMatrix<Type>::SymetriseParleBas()
{
   ELISE_ASSERT(_tx==_ty,"Non carree dans ::SymetriseParleBas");
    for (INT y =0; y<_ty; y++)
        for (INT x =0; x<y; x++)
              _data[y][x] = _data[x][y];
}



template <class Type>  void  ElMatrix<Type>::self_transpose ()
{
    ELISE_ASSERT(_tx==_ty,"Not a Square Matr in self_transpose");

    for (INT x =0; x<_tx; x++)
        for (INT y =0; y<x; y++)
            ElSwap(_data[y][x],_data[x][y]);
}


template <class Type>  ElMatrix<Type>  ElMatrix<Type>::transpose() const
{
     ElMatrix<Type> res(_ty,_tx);
     res.transpose(*this);
     return res;
}


template <class Type>
          Type  ElMatrix<Type>::L2(const ElMatrix<Type> & m2) const
{
    Type res =0;
    for (INT x =0; x<_tx; x++)
        for (INT y =0; y<_ty; y++)
              res = res+ ElSquare(_data[y][x]-m2._data[y][x]);
    return res;
}

template <class Type>
          Type  ElMatrix<Type>::scal(const ElMatrix<Type> & m2) const
{
    Type res =0;
    for (INT x =0; x<_tx; x++)
        for (INT y =0; y<_ty; y++)
              res = res+ _data[y][x]*m2._data[y][x];
    return res;
}



template <class Type>  Type  ElMatrix<Type>::L2() const
{
    Type res =0;
    for (INT x =0; x<_tx; x++)
        for (INT y =0; y<_ty; y++)
              res = res+ ElSquare(_data[y][x]);
    return res;
}

template <class Type> Type  ElMatrix<Type>::Det() const
{
   ELISE_ASSERT(_tx==_ty,"Not a Square Mat in Det");

   if (_tx == 1)
      return _data[0][0];

   if (_tx == 2)
      return _data[0][0] *  _data[1][1] - _data[1][0] *  _data[0][1] ;

   if (_tx == 3)
      return
              _data[0][0] * (_data[1][1] *  _data[2][2] - _data[1][2] *  _data[2][1])
            - _data[1][0] * (_data[0][1] *  _data[2][2] - _data[0][2] *  _data[2][1])
            + _data[2][0] * (_data[0][1] *  _data[1][2] - _data[0][2] *  _data[1][1]) ;


   ELISE_ASSERT(false,"ElMatrix Det , dim >3");
   return 244;

}

template <class Type> Type  ElMatrix<Type>::Trace() const
{
   ELISE_ASSERT(_tx==_ty,"Not a Square Mat in Det");
   Type aRes = 0;

   for (INT aXY=0 ; aXY<_tx ; aXY++)
       aRes += _data[aXY][aXY];
   return aRes;
}


template <class Type> void ElMatrix<Type>::set_shift_mat_permut(INT ShiftPremierCol)
{
   ELISE_ASSERT(_tx==_ty,"Not a Square Mat in Det");

   for (INT y=0; y< _tx ; y++)
   {
       for (INT x=0; x< _tx ; x++)
       {
          INT V = (y==mod(x+ShiftPremierCol,_tx)) ? 1 : 0;
          _data[y][x] = V;

       }
   }
}
template <class Type>  ElMatrix<Type> ElMatrix<Type>::transposition(INT aN,INT aK1,INT aK2)
{
   ElMatrix<Type> aRes(aN,true);

   for (INT y=0; y< aN ; y++)
       for (INT x=0; x< aN ; x++)
       {
          aRes._data[y][x] = (x==y) ? 1 : 0;
       }

   aRes._data[aK1][aK1] = aRes._data[aK2][aK2] = 0;
   aRes._data[aK1][aK2] = aRes._data[aK2][aK1] = 1;

   return aRes;
}

template <class Type>
ElMatrix<Type> ElMatrix<Type>::sub_mat(INT aCol, INT aLig, INT aNbCol, INT aNbLig) const
{
   ELISE_ASSERT(aCol<_tx,"aCol out of bound");
   ELISE_ASSERT(aLig<_ty,"aLig out of bound");
   ELISE_ASSERT(aNbCol<=_tx,"aNbCol out of bound");
   ELISE_ASSERT(aNbLig<=_ty,"aNbLig out of bound");

   ElMatrix<Type> aRes(aNbCol, aNbLig);

   int aK=0;
   int bK=0;
   for (INT y=aLig; y< aLig+aNbLig ; y++, bK++)
   {
       aK=0;
       for (INT x=aCol; x< aCol+aNbCol ; x++, aK++)
       {
           //std::cout << "aK, bK : " << aK << "  " << bK << endl;
           //std::cout << "x, y : " << x << "  " << y << endl;
           aRes(aK,bK) = (*this)(x,y);
       }
   }

   return aRes;
}

template <class Type>  ElMatrix<Type>
     ElMatrix<Type>::ExtensionId (INT ExtAvant,INT ExtApres) const
{
   ELISE_ASSERT(_tx==_ty,"Not a Square Mat in Det");
   INT Tres = _tx +ExtAvant + ExtApres;
   ElMatrix<Type> aRes(Tres,true);

   for (INT y=0; y< _tx ; y++)
       for (INT x=0; x< _tx ; x++)
           aRes(x+ ExtAvant,y+ ExtAvant) = (*this)(x,y);

   return aRes;
}


/*************************************************************/
/*                                                           */
/*             ROTATIONS                                     */
/*                                                           */
/*************************************************************/

template <class Type>
         ElMatrix<Type>   ElMatrix<Type>::Rotation(INT sz,Type teta,INT k1,INT k2)
{
//std::cout << "K1K2 " << k1 << " " << k2 << "\n";
     ElMatrix<Type> res(sz,true);
     res._data[k1][k1] =  (Type) cos(teta);
     res._data[k2][k1] =  (Type) sin(teta);
     res._data[k1][k2] =  (Type) -sin(teta);
     res._data[k2][k2] =  (Type) cos(teta);

     return res;
}

template <class Type>
         ElMatrix<Type>   ElMatrix<Type>::Rotation3D(Type teta,INT aNumAxeInv)
{
//std::cout << "Rotation3D " << aNumAxeInv << "\n";
   return Rotation(3,teta,(aNumAxeInv+1)%3,(aNumAxeInv+2)%3);
}

template <class Type>
         ElMatrix<Type>   ElMatrix<Type>::DerRotation
                          (INT sz,Type teta,INT k1,INT k2)
{
     ElMatrix<Type> res(sz,sz,0);
     res._data[k1][k1] =  (Type) -sin(teta);
     res._data[k2][k1] =  (Type) cos(teta);
     res._data[k1][k2] =  (Type) -cos(teta);
     res._data[k2][k2] =  (Type) -sin(teta);

     return res;
}

template <class Type> ElMatrix<Type>
         ElMatrix<Type>::Rotation
         (
             Pt3d<Type> aImI,
             Pt3d<Type> aImJ,
             Pt3d<Type> aImK
         )
{
   ElMatrix<Type> aMat = MatFromCol(aImI,aImJ,aImK);
   aMat.ColSchmidtOrthog();
   return aMat;
}

template <class Type>
       void   ElMatrix<Type>::PermRot(const std::string & aName,tTab3P  & aV)
{
    const char * aC = aName.c_str();
    for (int aK=0 ; aK<3 ; aK++)
    {
       aV[aK] = Pt3d<Type>(0,0,0);
       Type aSign = 1;
       if ((*aC==0) && (aK==2))
       {
           aV[2] = aV[0] ^ aV[1];
       }
       else
       {
           if (*aC=='-')
           {
               aSign = -1;
               aC++;
           }
           switch (*aC)
           {
               case 'i' : case 'I' : case 'x' : case 'X' :
                   aV[aK].x = aSign;
               break;
               case 'j' : case 'J' : case 'y' : case 'Y' :
                   aV[aK].y = aSign;
               break;
               case 'k' : case 'K' : case 'z' : case 'Z' :
                   aV[aK].z = aSign;
               break;

               default :
               {
                  std::cout << "FOR : " << aName << "\n";
                  ELISE_ASSERT(false,"invalid string in ElMatrix<Type>::PermRot");
               }
           }
           aC++;
       }
    }
    if (*aC!=0)
    {
        std::cout << "FOR : " << aName << "\n";
        ELISE_ASSERT(false,"invalid string in ElMatrix<Type>::PermRot");
    }

}

template <class Type>
       ElMatrix<Type>   ElMatrix<Type>::PermRot(const std::string & aName)
{
    Pt3d<Type>  aV[3];
    PermRot(aName,aV);
    return Rotation(aV[0],aV[1],aV[2]);
}


template <class Type>
       ElMatrix<Type>   ElMatrix<Type>::Rotation(Type teta01,Type teta02,Type teta12)
{
    return
              Rotation(3,teta01,0,1)
            * Rotation(3,teta02,0,2)
            * Rotation(3,teta12,1,2) ;
}

template <class Type>
       ElMatrix<Type>   ElMatrix<Type>::DDteta01
                        (Type teta01,Type teta02,Type teta12)
{
    return
              DerRotation(3,teta01,0,1)
            * Rotation(3,teta02,0,2)
            * Rotation(3,teta12,1,2) ;
}
template <class Type>
       ElMatrix<Type>   ElMatrix<Type>::DDteta02
                        (Type teta01,Type teta02,Type teta12)
{
    return
              Rotation(3,teta01,0,1)
            * DerRotation(3,teta02,0,2)
            * Rotation(3,teta12,1,2) ;
}
template <class Type>
       ElMatrix<Type>   ElMatrix<Type>::DDteta12
                        (Type teta01,Type teta02,Type teta12)
{
    return
              Rotation(3,teta01,0,1)
            * Rotation(3,teta02,0,2)
            * DerRotation(3,teta12,1,2) ;
}

ElMatrix<REAL> DiffRotEn1Pt(REAL teta01,REAL teta02,REAL teta12,Pt3dr pt)
{
    ElMatrix<REAL> M01 = ElMatrix<REAL>::Rotation(3,teta01,0,1);
    ElMatrix<REAL> D01 = ElMatrix<REAL>::DerRotation(3,teta01,0,1);
    ElMatrix<REAL> M02 = ElMatrix<REAL>::Rotation(3,teta02,0,2);
    ElMatrix<REAL> D02 = ElMatrix<REAL>::DerRotation(3,teta02,0,2);
    ElMatrix<REAL> M12 = ElMatrix<REAL>::Rotation(3,teta12,1,2);
    ElMatrix<REAL> D12 = ElMatrix<REAL>::DerRotation(3,teta12,1,2);

    return  MatFromCol
            (
               D01 * (M02 * (M12 * pt)),
               M01 * (D02 * (M12 * pt)),
               M01 * (M02 * (D12 * pt))
            );
}





void StdAngleFromRot
     (
        const ElMatrix<REAL> & m,
        REAL & a,REAL & b,REAL & c,
        REAL & score,
        INT sign
     )
{
    REAL sinb  = m(0,2);
    REAL cosb = sign * sqrt(1-ElSquare(sinb));

    REAL cosa = m(0,0) / cosb;
    REAL sina = m(0,1) / cosb;


    REAL cosc = m(2,2) / cosb;
    REAL sinc = m(1,2) / cosb;



    REAL  A = atan2(sina,cosa);
    REAL  B = atan2(sinb,cosb);
    REAL  C = atan2(sinc,cosc);

    // REAL new_score = m.L2(ElMatrix<REAL>::Rotation(A,B,C));
    REAL new_score = ElAbs(B);

    if (new_score < score)
    {
        score = new_score;
        a = A;
        b = B;
        c = C;
    }
}

/*
static REAL ToStdAng(REAL a)
{
    while (a>  PI) a -= 2* PI;
    while (a< -PI) a += 2* PI;
    return a;
}
*/


void AngleFromRot(const ElMatrix<REAL> & m,REAL & a,REAL & b,REAL & c)
{
    static const REAL epsilon = 1e-5;

    REAL sinb  = m(0,2);

    if (ElAbs(sinb) < 1-epsilon)
    {
        REAL score = 1e5;
        StdAngleFromRot(m,a,b,c,score,1);
        StdAngleFromRot(m,a,b,c,score,-1);

/*
        if (ElAbs(b) > PI/2)
        {
cout << "CHHHHHA\n";
            a = ToStdAng(a-PI);
            b = ToStdAng(b-PI);
            c = ToStdAng(c-PI);
        }
*/
    }
    else
    {
         b = (sinb>0) ? PI/2.0 : (-PI/2);
         c = 0;
         a = atan2(-m(1,0),m(1,1));
    }
}

void AngleFromRot(const ElMatrix<Fonc_Num> & ,Fonc_Num & ,Fonc_Num & ,Fonc_Num & )
{
    ELISE_ASSERT(false,"NO AngleFromRot for Fonc_Num");
}


template <class Type>   Type ElMatrix<Type>::NormC(INT x) const
{
    return ProdCC(*this,x,x);
}


template <class Type> void ElMatrix<Type>::SetColSchmidtOrthog(INT NbIter)
{
//std::cout << "::SetColSchmidtOrthoguuu\n";
//getchar();
    ELISE_ASSERT(_tx<=_ty,"Bad size in ElMatrix::ColSchmidtOrthog");

    for (INT x=0; x<_tx ; x++)
    {
        for (INT it =0; it<NbIter ; it++)
        {
            for (INT xp =0; xp<x ; xp++)
            {
                   Type s = ProdCC(*this,xp,x);
                   for(INT y =0; y< _ty ; y++)
                       _data[y][x] = _data[y][x] - _data[y][xp]*s;
            }
            Type N = (Type)sqrt(NormC(x));

            for(INT y =0; y< _ty ; y++)
                _data[y][x] = _data[y][x]/N;
        }
    }
}

template <class Type>
         ElMatrix<Type> ElMatrix<Type>::ColSchmidtOrthog(INT iter) const
{
   ElMatrix<Type> res (*this);
   res.SetColSchmidtOrthog(iter);
   return res;
}

void SauvFile(const ElRotation3D & aRot,const std::string & aName)
{
    ELISE_fp aFile(aName.c_str(),ELISE_fp::WRITE);
    aFile.write(aRot);
    aFile.close();
}


void XML_SauvFile(const ElRotation3D & aRC2M,const std::string & aName,const std::string & aNameEngl,bool aModeMatr)
{
    cOrientationExterneRigide anOER = From_Std_RAff_C2M(aRC2M,aModeMatr);
    MakeFileXML(anOER,aName,aNameEngl);
}

void XML_SauvFile(const ElRotation3D & aRC2M,const std::string & aName,const std::string & aNameEngl,double altisol,double prof,bool aModeMatr)
{
    cOrientationExterneRigide anOER = From_Std_RAff_C2M(aRC2M,aModeMatr);
    anOER.AltiSol().SetVal(altisol);
    anOER.Profondeur().SetVal(prof);
    MakeFileXML(anOER,aName,aNameEngl);
}



ElRotation3D ReadFromFile(const ElRotation3D *,const std::string & aName)
{
    ELISE_fp aFile(aName.c_str(),ELISE_fp::READ);
    ElRotation3D aRot = aFile.read((ElRotation3D *)0);
    aFile.close();
    return aRot;
}

double ProfFromCam(const ElRotation3D & anOr,const Pt3dr & aP)
{
   CamStenopeIdeale aCam =CamStenopeIdeale::CameraId(true,anOr);
   return aCam.ProfondeurDeChamps(aP);

}

ElMatrix<REAL>  VectRotationArroundAxe(const Pt3dr & aV00,double aTeta)
{
  Pt3dr aV0 = aV00;
  Pt3dr aV1,aV2;
  MakeRONWith1Vect(aV0,aV1,aV2);

  return ComplemRotation(aV0,aV1,aV0,aV1*cos(aTeta) + aV2*sin(aTeta));

}

ElRotation3D RotationOfInvariantPoint(const Pt3dr & aP0 ,const ElMatrix<double> & aMat)
{
    return ElRotation3D(aP0 - aMat*aP0,aMat,true);
}

ElRotation3D  AffinRotationArroundAxe(const ElSeg3D & aSeg,double aTeta)
{
    return RotationOfInvariantPoint(aSeg.P0(), VectRotationArroundAxe(aSeg.Tgt(),aTeta));
}


/*
ElMatrix<REAL>  VectRotationArroundAxe(const Pt3dr &,double aTeta);
ElRotation3D  AffinRotationArroundAxe(const ElSeg3D &,double aTeta);
ElRotation3D RotationOfInvariantPoint(const Pt3dr & ,const ElMatrix<double> &);
*/


//  Q1 = tr1 + Mat1 * aP1
//  X2 = S2toS1(X1) = S2.FromSys2This(S1,X1)
//  X1 = S1.FromSys2This(S2,X2)

/*
ElRotation3D  ChangementSysC
              (
                     const Pt3dr &       aP1,
                     const ElRotation3D& aR1,
                     const cSysCoord & a1Source,
                     const cSysCoord & a2Cible
              )
{
     std::cout << "NON TESTE !! \n";
     Pt3dr aP2 = a2Cible.FromSys2This(a1Source,aP1);
     ElMatrix<double>  aMatr2 =      a2Cible.JacobSys2This(a1Source,aP1)
                                  * aR1.Mat()
                                  * a1Source.JacobSys2This(a2Cible,aP2);

     aMatr2 = NearestRotation(aMatr2);

     Pt3dr aQ1 = aR1.ImAff(aP1);
     Pt3dr aQ2 = a2Cible.FromSys2This(a1Source,aQ1);


     Pt3dr aTr2 = aQ2 - aMatr2 * aP2;

     return ElRotation3D(aTr2,aMatr2);
}
*/


/*************************************************************/
/*                                                           */
/*             INSTANCIATION                                 */
/*                                                           */
/*************************************************************/

template class ElMatrix<INT>;
template class ElMatrix<REAL>;
template class ElMatrix<REAL16>;
template class ElMatrix<Fonc_Num>;


#define INST_MAT_SCAL(Type)\
template Pt2d<Type> mul32 (const ElMatrix<Type> & M,const Pt3d<Type> &p);\
template void SetCol (ElMatrix<Type> & M,INT col,Pt3d<Type> );\
template void SetLig (ElMatrix<Type> & M,INT Lig,Pt3d<Type> );\
template void SetCol (ElMatrix<Type> & M,INT col,Pt2d<Type> );\
template void SetLig (ElMatrix<Type> & M,INT Lig,Pt2d<Type> );\
template Pt2d<Type> operator * (const ElMatrix<Type> & M,const Pt2d<Type> &p);\


template Pt3d<INT> operator * (const ElMatrix<INT> & M,const Pt3d<INT> &p);

template ElMatrix<INT> MatFromCol (Pt3d<INT>,Pt3d<INT>,Pt3d<INT>);
template ElMatrix<REAL> MatFromCol (Pt3d<REAL>,Pt3d<REAL>,Pt3d<REAL>);

template ElMatrix<REAL> MatFromCol (Pt2d<REAL>,Pt2d<REAL>);
template ElMatrix<REAL16> MatFromCol (Pt2d<REAL16>,Pt2d<REAL16>);
template ElMatrix<Fonc_Num> MatFromCol (Pt2d<Fonc_Num>,Pt2d<Fonc_Num>);

//   X     A     YC - BZ      0 -Z  Y    A
//   Y  ^  B  =  ZA - XC   =  Z  0 -X  * B
//   Z     C     XB - YA      -Y X  0    C

ElMatrix<REAL>  MatProVect(const Pt3dr & aP)
{
   ElMatrix<REAL> aRes (3,3);
   aRes(0,0) = 0;
   aRes(1,0) = -aP.z;
   aRes(2,0) =  aP.y;
   aRes(0,1) = aP.z;
   aRes(1,1) = 0;
   aRes(2,1) = -aP.x;
   aRes(0,2) = -aP.y;
   aRes(1,2) = aP.x,
   aRes(2,2) = 0.0;

   return aRes;
}



INST_MAT_SCAL(INT)
INST_MAT_SCAL(REAL)
INST_MAT_SCAL(Fonc_Num)
INST_MAT_SCAL(REAL16)

#define INST_ALL_MAT(Type)\
template ElMatrix<Type> operator * (const Type &,const ElMatrix<Type>&);


INST_ALL_MAT(INT);
INST_ALL_MAT(REAL);
INST_ALL_MAT(REAL16);
INST_ALL_MAT(Fonc_Num);


template  ElMatrix<REAL>   gaussj(const ElMatrix<REAL> & m);
template  void   self_gaussj(ElMatrix<REAL> & m);
template  bool   self_gaussj_svp(ElMatrix<REAL> & m);


template  ElMatrix<REAL16>   gaussj(const ElMatrix<REAL16> & m);
template  void   self_gaussj(ElMatrix<REAL16> & m);
template  bool   self_gaussj_svp(ElMatrix<REAL16> & m);


/*void F()
{
    ElMatrix<REAL> M(2,2);
    M = gaussj(M);
    self_gaussj(M);
    self_gaussj_svp(M);
}*/

/*************************************************************/
/*                                                           */
/*             ElRotation3D                                  */
/*                                                           */
/*************************************************************/


ElMatrix<double> InvMatrix(const ElMatrix<double> & mat)
{
    return gaussj(mat);
}

ElMatrix<Fonc_Num> InvMatrix(const ElMatrix<Fonc_Num> & mat)
{
    ELISE_ASSERT(false,"InvMatrix(const ElMatrix<Fonc_Num> & mat)");
    return mat;
}


template <class Type> TplElRotation3D<Type>::TplElRotation3D(Pt3d<Type> tr,const ElMatrix<Type> & mat,bool aTrueRot) :
    _tr       (tr),
    _Mat      ( aTrueRot ? mat.ColSchmidtOrthog() : mat),
    _InvM     (aTrueRot  ? _Mat.transpose()       : InvMatrix(mat)),
    mTrueRot  (aTrueRot)
{
     if (aTrueRot)
     {
        AngleFromRot(_Mat,_teta01,_teta02,_teta12);
     }
     else
     {
          _teta01= strtod("NAN(teta01)", NULL);
          _teta02= strtod("NAN(teta02)", NULL);
          _teta12= strtod("NAN(teta12)", NULL);
     }
}



template <class Type> TplElRotation3D<Type>::TplElRotation3D (Pt3d<Type> tr,Type teta01,Type teta02,Type teta12) :
    _tr     (tr),
    _Mat    (ElMatrix<Type>::Rotation(teta01,teta02,teta12)),
    _InvM   (_Mat.transpose()),
    _teta01 (teta01),
    _teta02 (teta02),
    _teta12 (teta12),
    mTrueRot (true)
{
}


template <class Type>  void  TplElRotation3D<Type>::AssertTrueRot() const
{
   if (AcceptFalseRot)
   {
        cElWarning::TrueRot.AddWarn("Accept False Rot",__LINE__,__FILE__);
   }
   else
   {
       ELISE_ASSERT(mTrueRot,"Expecting true rotation");
   }
}


template <class Type>   TplElRotation3D<Type> &
                        TplElRotation3D<Type>::operator = (const  TplElRotation3D<Type> & aR2)
{
   if (this == & aR2)
       return *this;

   _tr = aR2._tr;
   _Mat = aR2._Mat;
   _InvM = aR2._InvM;
   _teta01 = aR2._teta01;
   _teta02 = aR2._teta02;
   _teta12 = aR2._teta12;
   mTrueRot = aR2.mTrueRot;
   return *this;
}

template <>  TplElRotation3D<REAL> TplElRotation3D<REAL>::inv() const
{
   return ElRotation3D ( -(_InvM*_tr), _InvM,mTrueRot);
}

template <> TplElRotation3D<Fonc_Num> TplElRotation3D<Fonc_Num>::inv() const
{
   return TplElRotation3D<Fonc_Num>( -(_InvM*_tr),-_teta12,-_teta02,-_teta01);
}



template <class Type>  TplElRotation3D<Type>
                       TplElRotation3D<Type>::operator *(const TplElRotation3D<Type> & R2) const
{
   return TplElRotation3D<Type>
          (
              _tr+_Mat*R2._tr,
              _Mat*R2._Mat,
              mTrueRot && R2.mTrueRot
          );
}


template <class Type> bool TplElRotation3D<Type>::IsTrueRot() const
{
    return mTrueRot;
}

template <class Type> Pt3d<Type> TplElRotation3D<Type>:: ImAff( Pt3d<Type> p) const
{
    return _tr + _Mat * p;
}
template <class Type> Pt3d<Type> TplElRotation3D<Type>::ImRecAff( Pt3d<Type> p) const
{
    return _InvM * (p-_tr);
}
template <class Type> Pt3d<Type> TplElRotation3D<Type>::ImVect( Pt3d<Type> p) const
{
    return  _Mat * p;
}
template <class Type> Pt3d<Type> TplElRotation3D<Type>::IRecVect( Pt3d<Type> p) const
{
    return _InvM * p;
}


template <> ElMatrix<REAL>  TplElRotation3D<REAL>::DiffParamEn1pt(Pt3dr p) const
{
    ElMatrix<REAL> res(6,3);

    ElMatrix<REAL> Mteta = DiffRotEn1Pt(_teta01,_teta02,_teta12,p);
    for (int x=0; x<3 ; x++)
        for (int y=0; y<3 ; y++)
        {
             res(x,y) = (x==y);
             res(x+3,y) = Mteta(x,y);
        }

    return res;
}




template class TplElRotation3D<REAL>;
template class TplElRotation3D<Fonc_Num>;

#define InstantieId(Type)\
template <> const TplElRotation3D<Type> TplElRotation3D<Type>::Id(Pt3d<Type>(0.0,0.0,0.0),Type(0.0),Type(0.0),Type(0.0));

InstantieId(REAL)

/*

         Pt3dr operator()(Pt3dr);

*/


// Un peu bovin mais sinon ne passe pas les test de fuite memoire


void ComplBaseParLeHaut(ElMatrix<REAL> &aM,INT NbLigneOk)
{
    ELISE_ASSERT(aM.tx() >= aM.ty(),"Size in ComplBaseParLeHaut");
    ElMatrix<REAL>  mNormCBPH(1,aM.ty());

    for (INT y =0 ; y<aM.ty(); y++)
    {
        mNormCBPH(0,y) = 1.0;
    }
    for (INT y = aM.ty()-NbLigneOk; y<aM.ty() ; y++)
    {
        mNormCBPH(0,y) = 0.0;
        for (int x=0; x<aM.tx() ; x++)
           mNormCBPH(0,y)+= ElSquare(aM(x,y));
        mNormCBPH(0,y) = sqrt(mNormCBPH(0,y));
    }

    for (INT IndL = aM.ty()-NbLigneOk-1; IndL>=0 ;  IndL--)
    {
         INT xMinMax = -1;
     REAL aCosMinMax = 2.0;
     for (INT x =0 ; x<aM.tx() ; x++)
     {
         REAL aCosMax = -1.0;
         for (int y=IndL+1; y<aM.ty() ; y++)
         {
                      REAL aCos = aM(x,y)/mNormCBPH(0,y);
                      ElSetMax(aCosMax,ElAbs(aCos));
         }
         ELISE_ASSERT(aCosMax>=0,"ComplBaseParLeHaut");
         if (aCosMax < aCosMinMax)
         {
             aCosMinMax = aCosMax;
             xMinMax  =x ;
         }
     }
     ELISE_ASSERT(xMinMax!=-1,"ComplBaseParLeHaut");
     for (INT x =0 ; x<aM.tx() ; x++)
         aM(x,IndL) = (x==xMinMax);
    }
}


// ====================================================
//
//    cChCoCart
//
// ====================================================

cChCoCart::cChCoCart
(
    const Pt3dr& aOri,
    const Pt3dr& anOx,
    const Pt3dr& anOy,
    const Pt3dr& anOz
) :
   mOri (aOri),
   mOx  (anOx),
   mOy  (anOy),
   mOz  (anOz)
{
}

Pt3dr cChCoCart::FromLoc(const Pt3dr & aP) const
{
   return mOri + mOx*aP.x + mOy*aP.y + mOz*aP.z;
}

cChCoCart cChCoCart::Inv() const
{
  ElMatrix<double> aM = gaussj(MatFromCol(mOx,mOy,mOz));

   Pt3dr aOx,aOy,aOz;
   aM.GetCol(0,aOx);
   aM.GetCol(1,aOy);
   aM.GetCol(2,aOz);

  return cChCoCart ( -(aM*mOri),aOx,aOy,aOz);
}


cChCoCart cChCoCart::Xml2El(const cRepereCartesien & aRep)
{
  return cChCoCart(aRep.Ori(),aRep.Ox(),aRep.Oy(),aRep.Oz());
}

cRepereCartesien cChCoCart::El2Xml() const
{
   cRepereCartesien aRes;
   aRes.Ori() = mOri;
   aRes.Ox()  = mOx;
   aRes.Oy()  = mOy;
   aRes.Oz()  = mOz;

   return aRes;
}









/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
