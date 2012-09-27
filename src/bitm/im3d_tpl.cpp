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






/***********************************************************************/
/*                                                                     */
/*       DataIm3D                                                      */
/*                                                                     */
/***********************************************************************/

template <class Type,class TyBase>
     const INT *  DataIm3D<Type,TyBase>::p0() const
{
     return PTS_00000000000000;
}


template <class Type,class TyBase>
     const INT *  DataIm3D<Type,TyBase>::p1() const
{
   return (_txyz);
}





template <class Type,class TyBase> 
         DataIm3D<Type,TyBase>::DataIm3D
         (
	     INT Tx,INT Ty,INT Tz,
	     bool to_init,TyBase v_init,const char * str_init,
	     Type * DataLin
	 ) :
         DataGenImType<Type,TyBase>((DataLin==0) ? Tx*Ty*Tz : 0,to_init,v_init,str_init)
{
    if (DataLin)
    {
        this->_data_lin = DataLin;
        this->_to_free = false;
    }

    _txyz[0] = Tx;
    _txyz[1] = Ty;
    _txyz[2] = Tz;
    _data = NEW_MATRICE(Pt2di(0,0),Pt2di(Ty,Tz),Type *);
    for (INT z=0; z<Tz; z++)
        for (INT y=0; y<Ty; y++)
            _data[z][y] = this->_data_lin + Tx*(y+z*Ty);
}

template <class Type,class TyBase> INT DataIm3D<Type,TyBase>::dim() const
{
   return 3;
}






template <class Type,class TyBase> 
          void  * DataIm3D<Type,TyBase>::calc_adr_seg(INT * pts)
{
    return _data[pts[2]][pts[1]] ;
}


template <class Type,class TyBase> DataIm3D<Type,TyBase>::~DataIm3D()
{
     if (_data)
     {
         DELETE_MATRICE(_data,Pt2di(0,0),Pt2di(_txyz[1],_txyz[2]));
         _data = 0;
     }
}




template <class Type,class TyBase> void DataIm3D<Type,TyBase>::out_pts_integer 
              (Const_INT_PP pts,INT nb,const void * i)
{
   const INT * tx = pts[0];
   const INT * ty = pts[1];
   const INT * tz = pts[2];
   const TyBase * in =  C_CAST(const TyBase *, i);

   for (int j=0 ; j<nb ; j++)
       _data[tz[j]][ty[j]][tx[j]] = (Type) in[j];
}


template <class Type,class TyBase> void DataIm3D<Type,TyBase>::input_pts_integer 
              (void * o,Const_INT_PP pts,INT nb) const
{
   const INT * tx = pts[0];
   const INT * ty = pts[1];
   const INT * tz = pts[2];
   TyBase * out =  C_CAST(TyBase *,o);

   for (int i=0 ; i<nb ; i++)
       out[i] =  _data[tz[i]][ty[i]][tx[i]];
}


template <class Type,class TyBase> void DataIm3D<Type,TyBase>::input_pts_reel
              (REAL * out,Const_REAL_PP pts,INT nb) const
{
   const REAL * tx = pts[0];
   const REAL * ty = pts[1];
   const REAL * tz = pts[2];

   REAL x,y,z;
   REAL p_0x,p_1x,p_0y,p_1y,p_0z,p_1z;
   INT xi,yi,zi;

   for (int i=0 ; i<nb ; i++)
   {
       x = tx[i];
       y = ty[i];
       z = tz[i];
       p_1x = x - (xi= (INT) x);
       p_1y = y - (yi= (INT) y);
       p_1z = z - (zi= (INT) z);
       p_0x = 1.0-p_1x;
       p_0y = 1.0-p_1y;
       p_0z = 1.0-p_1z;

       out[i] =

             p_0z *
             (   p_0x * p_0y * _data[ zi ][ yi ][ xi ]
               + p_1x * p_0y * _data[ zi ][ yi ][xi+1]
               + p_0x * p_1y * _data[ zi ][yi+1][ xi ]
               + p_1x * p_1y * _data[ zi ][yi+1][xi+1]
             )
         +   p_1z * 
             (   p_0x * p_0y * _data[zi+1][ yi ][ xi ]
               + p_1x * p_0y * _data[zi+1][ yi ][xi+1]
               + p_0x * p_1y * _data[zi+1][yi+1][ xi ]
               + p_1x * p_1y * _data[zi+1][yi+1][xi+1]
             );
   }
}



template <class Type,class TyBase>
         void DataIm3D<Type,TyBase>::out_assoc
         (
                  void * out, 
                  const OperAssocMixte & op,
                  Const_INT_PP coord,
                  INT nb,
                  const void * values
         )
         const
{
      TyBase * v = (TyBase *) const_cast<void *>(values);
      TyBase * o =  (TyBase *) out;
      const INT * x    = coord[0];
      const INT * y    = coord[1];
      const INT * z    = coord[2];

      Type  * adr_vxy;
      INT nb0 = nb;

      if (op.id() == OperAssocMixte::Sum)
          while(nb--) 
          {
               adr_vxy = _data[*(z++)][*(y++)] + *(x++);
               *(o ++) =  *adr_vxy;
               *adr_vxy += (Type) *(v++);
          }
      else
          while(nb--)
          {
               adr_vxy = _data[*(z++)][*(y++)] + *(x++);
               *(o ++) =  *adr_vxy;
               *adr_vxy = (Type)op.opel((TyBase)(*adr_vxy),*(v++));
          }
      verif_value_op_ass(op,o-nb0,v-nb0,nb0,(TyBase)this->v_min,(TyBase)this->v_max);
}



template <class Type,class TyBase>
    Type ***  DataIm3D<Type,TyBase>::data() const {return _data;}



template <class Type,class TyBase>
    INT DataIm3D<Type,TyBase>::tx() const {return _txyz[0];}

template <class Type,class TyBase>
    INT DataIm3D<Type,TyBase>::ty() const {return _txyz[1];}

template <class Type,class TyBase>
    INT DataIm3D<Type,TyBase>::tz() const {return _txyz[2];}





           /***********************/
           /*    Im3D             */
           /***********************/

template <class Type,class TyBase> INT   Im3D<Type,TyBase>::vmax() const
{
      return ((DataIm3D<Type,TyBase> *) (_ptr))->v_max;
}


template <class Type,class TyBase> 
        Im3D<Type,TyBase>::Im3D(INT tx,INT ty,INT tz) :
        GenIm (new DataIm3D<Type,TyBase> (tx,ty,tz,false,0))
{
}

template <class Type,class TyBase> 
        Im3D<Type,TyBase>::Im3D(INT tx,INT ty,INT tz,TyBase v_init) :
        GenIm (new DataIm3D<Type,TyBase> (tx,ty,tz,true,v_init))
{
}

template <class Type,class TyBase> 
        Im3D<Type,TyBase>::Im3D(INT tx,INT ty,INT tz,const char * v_init) :
        GenIm (new DataIm3D<Type,TyBase> (tx,ty,tz,false,0,v_init))
{
}

template <class Type,class TyBase> 
        Im3D<Type,TyBase>::Im3D
        (
	         Im3D_WithDataLin,
                 INT tx,
                 INT ty,
                 INT tz,
                 Type * DataLin
        ) :
        GenIm (new DataIm3D<Type,TyBase> (tx,ty,tz,false,0,0,DataLin))
{
}





template <class Type,class TyBase> Type *** Im3D<Type,TyBase>::data()
{
      return ((DataIm3D<Type,TyBase> *) (_ptr))->data();
}


template <class Type,class TyBase> INT  Im3D<Type,TyBase>::tx() const
{
      return ((DataIm3D<Type,TyBase> *) (_ptr))->tx();
}

template <class Type,class TyBase> INT  Im3D<Type,TyBase>::ty() const
{
      return ((DataIm3D<Type,TyBase> *) (_ptr))->ty();
}

template <class Type,class TyBase> INT  Im3D<Type,TyBase>::tz() const
{
      return ((DataIm3D<Type,TyBase> *) (_ptr))->tz();
}

/***********************************************************************/
/***********************************************************************/
/***********************************************************************/
/***********************************************************************/

/*
#define INSTANTIATE_BITM_3D_GEN(Type,TyBase)\
{\
    Im3D<Type,TyBase> b3(2,3,4);\
    Im3D<Type,TyBase> b33(2,3,4,5);\
    b33 = Im3D<Type,TyBase> (2,3,4,"5");\
    cout << b3.data() << b3.tx() <<  b3.ty() << b3.tz() << b3.vmax();\
}

void instantiate_bitm_3d_gen()
{
}
*/



#define INSTANTIATE_BITM_3D_GEN(Type,TyBase)\
template class Im3D<Type,TyBase>;\
template class DataIm3D<Type,TyBase>;



INSTANTIATE_BITM_3D_GEN(U_INT1,INT);
INSTANTIATE_BITM_3D_GEN(INT1,INT);
INSTANTIATE_BITM_3D_GEN(U_INT2,INT);
INSTANTIATE_BITM_3D_GEN(INT2,INT);
INSTANTIATE_BITM_3D_GEN(INT4,INT);
INSTANTIATE_BITM_3D_GEN(REAL8,REAL8);
INSTANTIATE_BITM_3D_GEN(REAL4,REAL8);

void FiltrageImage3D
     (
	Im3D<double,double> Im,
	Pt2di               SzFiltr,
	INT                 Coord1,
	INT                 Coord2
     )
{
	INT Coord3 = 3-Coord1-Coord2;

	Fonc_Num F1 = kth_coord(Coord1);
	Fonc_Num F2 = kth_coord(Coord2);


	INT Sz1 = Im.P1()[Coord1];
	INT Sz2 = Im.P1()[Coord2];
	INT Sz3 = Im.P1()[Coord3];

	for (INT K=0 ; K<Sz3 ; K++)
	{
	   Fonc_Num FXYZ = Virgule(FX,FY,K);
	   if (Coord3 == 0)
              FXYZ = Virgule(K,FX,FY);
	   if (Coord3 == 1)
              FXYZ = Virgule(FX,K,FY);
	   ELISE_COPY
           (
	       rectangle(Pt2di(0,0),Pt2di(Sz1,Sz2)),
	         rect_som(Im.in(0)[FXYZ],SzFiltr) 
	       / rect_som(Im.inside()[FXYZ],SzFiltr),
	       Im.out().chc(FXYZ)
	   );
	}
}

void FiltrageImage3D
    (
         INT StepX,INT StepY, INT StepZ,
         double *  Data,
	 INT    Tx, INT    Ty, INT    Tz
    )
{
	Im3D_WithDataLin anObj;
	Im3D<double,double> Im(anObj,Tx,Ty,Tz,Data);

        FiltrageImage3D(Im,Pt2di(StepX,StepY),0,1);
        FiltrageImage3D(Im,Pt2di(StepX,StepZ),0,2);
        FiltrageImage3D(Im,Pt2di(StepY,StepZ),1,2);
}

void NoyauxFiltrageImage3D
    (
         INT StepX,INT StepY, INT StepZ,
         double *  Data,
	 INT    Tx, INT    Ty, INT    Tz
    )
{
     FiltrageImage3D(StepZ,StepY,StepX,Data,Tz,Ty,Tx);
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
