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


static const bool ContTile128 = false;

template <class Type> class Tile_UL_PCKB
{
	public :

		Tile_UL_PCKB(INT x0,INT x1);
		void UnInit()
		{
			if (buf) 
				DELETE_VECTOR(buf,0);
		}

		void UCompr
     	(
            ElSTDNS vector<U_INT2> &    LInd,
            ElSTDNS vector<U_INT2> &    VInd,
            ElSTDNS vector<U_INT1> &    Length,
            ElSTDNS vector<Type> &  	Vals
     	); 

		void UpdateSize(unsigned int sz)
		{
			ElSetMax(sz_buf,sz);
		}

		void init()
		{
			buf = NEW_VECTEUR(0,sz_buf,U_INT1);
		}
		void read (ELISE_fp &,tFileOffset nb);

        Type GetAValAndIncr()
        {
            Type aRes = El_CTypeTraits<Type>::RawData2Val(cur);
             cur +=  El_CTypeTraits<Type>::eSizeOf;
             return aRes;
        }
	
	private :
		U_INT1 * buf;	
		U_INT1 * cur;	
		unsigned int 	sz_buf;
		INT      nb_tot;
		bool     indexe ;
};


template <class Type> void Tile_UL_PCKB<Type>::UCompr
     (
         ElSTDNS vector<U_INT2> &    LInd,
         ElSTDNS vector<U_INT2> &    VInd,
         ElSTDNS vector<U_INT1> &    Length,
         ElSTDNS vector<Type> &    Vals
     )
{

    for (INT i=0; i<nb_tot; )
    {
        INT1 v;
		v = *((INT1 *)(cur++));
        if (v!= -128)
        {
            INT nb = (v>=0) ? (v+1) : (1-v);
            ELISE_ASSERT
            (
                (i+nb) <= nb_tot,
               "Incoherent chck_sum inPckBits"
            );
             if (v>=0)
             {
				Length.push_back(2*(nb-1));
				for (INT k=0; k<nb ; k++)
                {
					Vals.push_back(GetAValAndIncr());
                }
             }
             else
             {
				Length.push_back(2*(nb-1)+1);
			    Vals.push_back(GetAValAndIncr());
            }
           	i+= nb;
        }
        else
        {
        }                                    
	}

	if (indexe)
	{
		LInd.push_back((unsigned short) Length.size());
		VInd.push_back((unsigned short) Vals.size());
	}
}




template <class Type>  void Tile_UL_PCKB<Type>::read (ELISE_fp & fp,tFileOffset nb)
{
	fp.read(buf,sizeof(U_INT1),nb);
	cur = buf;
}

template <class Type>  Tile_UL_PCKB<Type>::Tile_UL_PCKB(INT x0,INT x1) :
	buf 	(0),
	sz_buf  (0),
	nb_tot	(x1-x0),
	// indexe	((x1%128) == 0)
	indexe	(true)
{
}


template <class Type> PackB_IM<Type> UnLoadPackBit<Type>::Do(DATA_Tiff_Ifd &aDTI,Tiff_Im::COMPR_TYPE aModeCompr)
{

	Tjs_El_User.ElAssert
	(
		aDTI._mode_compr == aModeCompr,
		EEM0 << "un_load_pack_bit handle only Pack Bits Images (!!) "
			<< "Tiff File = " << aDTI.name()
	);

	Tjs_El_User.ElAssert
	(
		aDTI._nb_chanel == 1,
		EEM0 << "un_load_pack_bit handle only mono chanel images "
			<< "Tiff File = " << aDTI.name()
	);

	Tjs_El_User.ElAssert
	(
		aDTI.type_el() == type_of_ptr((Type *)0),
		EEM0 << "un_load_pack_bit Got The Wrong Type"
			<< "Tiff File = " << aDTI.name()
	);

    if (ContTile128)  
    {
	    Tjs_El_User.ElAssert
	    (
		    (128%aDTI._sz_tile.x) == 0,
		    EEM0 << "un_load_pack_bit , sz of Dalle must divide 128"
			    << "Tiff File = " << aDTI.name()
	    );
    }


	ElSTDNS vector<Tile_UL_PCKB<Type> >  TUP;
	
	{
	for (INT Tilx =0,x=0; Tilx<aDTI._nb_tile.x; Tilx++,x+=aDTI._sz_tile.x)
	{
		TUP.push_back(Tile_UL_PCKB<Type>(x,x+aDTI._sz_tile.x));

		for (INT Tily =0; Tily<aDTI._nb_tile.y; Tily++)
		{
			Tjs_El_User.ElAssert
     		(
				 aDTI.offset_tile(Tilx,Tily,0) != Tiff_Im::UN_INIT_TILE,   
				EEM0 << "Use of uncomplete Tiff file"
			);
			TUP.back().UpdateSize(aDTI.byte_count_tile(Tilx,Tily,0).CKK_Byte4AbsLLO());
		}
		TUP.back().init();
	}
	}

	ELISE_fp  fp(aDTI.name(),ELISE_fp::READ);

   ElSTDNS vector<U_INT2>   LInd;
   ElSTDNS vector<U_INT2>   VInd;
   ElSTDNS vector<U_INT1>   Length;
   ElSTDNS vector<Type>  	Vals;

	Data_PackB_IM<Type> * res = new Data_PackB_IM<Type>(aDTI._sz.x,aDTI._sz.y,0,-aDTI._sz_tile.x);

	for (INT Tily =0; Tily<aDTI._nb_tile.y; Tily++)
	{
		for (INT Tilx =0; Tilx<aDTI._nb_tile.x; Tilx++)
		{
			fp.seek_begin(aDTI.offset_tile(Tilx,Tily,0));
			TUP[Tilx].read(fp,aDTI.byte_count_tile(Tilx,Tily,0));
		}
		INT y0 = Tily * aDTI._sz_tile.y;
		INT y1 = ElMin(y0+aDTI._sz_tile.y,aDTI._sz.y);

		for (INT y= y0; y<y1; y++)
		{
			LInd.clear(); LInd.push_back(0);
			VInd.clear(); VInd.push_back(0);
			Length.clear();
			Vals.clear();
			for (INT Tilx =0; Tilx<aDTI._nb_tile.x; Tilx++)
				TUP[Tilx].UCompr(LInd,VInd,Length,Vals);
			res->lpckb(y).init(0,LInd,VInd,Length,Vals);
		}
	}

	{
	for (INT Tilx =0; Tilx<aDTI._nb_tile.x; Tilx++)
		 TUP[Tilx].UnInit();
	}

        fp.close();
	return  PackB_IM<Type>(res);
	
}

PackB_IM<U_INT1> Tiff_Im::un_load_pack_bit_U_INT1()
{
	return UnLoadPackBit<U_INT1>::Do
           (
               *dtifd(),
               Tiff_Im::PackBits_Compr
           );
}

PackB_IM<U_INT2> Tiff_Im::un_load_pack_bit_U_INT2()
{
	return UnLoadPackBit<U_INT2>::Do
           (
               *dtifd(),
               Tiff_Im::NoByte_PackBits_Compr
           );
}





bool DATA_Tiff_Ifd::OkFor_un_load_pack_bit(INT Nbb,Tiff_Im::COMPR_TYPE aModeCompr)  
{
   return 
         (_mode_compr == aModeCompr)
      && (_nb_chanel == 1)
      && (_nbb_ch0 == Nbb)
      && ( (!ContTile128) || ((128%_sz_tile.x)==0));
}



bool Tiff_Im::OkFor_un_load_pack_bit_U_INT1()  
{
    return  dtifd()->OkFor_un_load_pack_bit(8,Tiff_Im::PackBits_Compr);
}

bool Tiff_Im::OkFor_un_load_pack_bit_U_INT2()  
{
    return  dtifd()->OkFor_un_load_pack_bit(16,Tiff_Im::NoByte_PackBits_Compr);
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
