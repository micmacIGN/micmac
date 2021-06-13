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
/*       DataGenIm                                                     */
/*                                                                     */
/***********************************************************************/

void DataGenIm::read_data(ELISE_fp & fp)
{
    fp.read(data_lin_gen(),sz_el(),sz_tot());
}

void DataGenIm::write_data(ELISE_fp & fp) const
{
    fp.write(const_cast<DataGenIm *>(this)->data_lin_gen(),sz_el(),sz_tot());
}





DataGenIm::~DataGenIm() {}


void DataGenIm::q_dilate
                   (  Std_Pack_Of_Pts<INT> * ,
                      char **,
                      const Std_Pack_Of_Pts<INT> * ,
                      INT **,
                      INT   ,
                      Image_Lut_1D_Compile   ,
                      Image_Lut_1D_Compile   
                   )
{
    elise_internal_error("Incoherent call to q_dilate",__FILE__,__LINE__);
}

void DataGenIm::verif_in_range_type 
    (INT* vals ,const Pack_Of_Pts * pts,INT v_max,INT v_min)
{
     if (El_User_Dyn.active() &&  (v_min != v_max))
     {
         INT index = index_values_out_of_range(vals,pts->nb(),v_min,v_max);

         if (index != INDEX_NOT_FOUND)
         {
           El_User_Dyn.ElAssert
           (
             index == INDEX_NOT_FOUND,
             EEM0  << "values out of range in bitmaps writing \n"
                   << "|   value = "   << vals[index]     << "\n"
                   << "|   pts =  "    << ElEM(pts,index)     << "\n"
                   << "|   interval = [" << v_min << " ---  " << v_max  << "["
           );
         }
     }
}

void DataGenIm::verif_in_range_type 
    (REAL * vals ,const Pack_Of_Pts * pts,INT v_max,INT v_min)
{
     if (El_User_Dyn.active() &&  (v_min != v_max))
     {
         INT index = index_values_out_of_range(vals,pts->nb(),(REAL)v_min,(REAL)v_max);

         if (index != INDEX_NOT_FOUND)
         {
		El_User_Dyn.ElAssert
         	(
             		index == INDEX_NOT_FOUND,
             		EEM0  << "values out of range in bitmaps writing \n"
                   		<< "|   value = "   << vals[index]     << "\n"
                   		<< "|   pts =  "    << ElEM(pts,index)     << "\n"
                   		<< "|   interval = [" << v_min << " ---  " << v_max  << "["
         	);
	}
     }
}



void DataGenIm::verif_in_range_type (INT* vals ,const Pack_Of_Pts * pts)
{
     verif_in_range_type(vals,pts,vmax(),vmin());
}

void DataGenIm::verif_in_range_type (REAL* vals ,const Pack_Of_Pts * pts)
{
     verif_in_range_type(vals,pts,vmax(),vmin());
}

void DataGenIm::tiff_predictor(INT ,INT ,INT,bool )
{
     El_Internal.ElAssert
     (
          0,
          EEM0 << "DataGenIm::tiff_predictor"
     );
}


      /***********/
      /*  GenIm  */
      /***********/


GenIm::GenIm(DataGenIm * gi) :
    PRC0(gi)
{
}

Fonc_Num  GenIm::in()
{
    return new
           ImInNotComp (SAFE_DYNC(DataGenIm *,_ptr),*this,false,0);
}

Fonc_Num  GenIm::in(REAL val)
{
    return new
           ImInNotComp(SAFE_DYNC(DataGenIm *,_ptr),*this,true,val);
}


Output GenIm::out()
{
    return new ImOutNotComp(SAFE_DYNC(DataGenIm *,_ptr),*this,true,false);
}

Output GenIm::oclip()
{
    return new ImOutNotComp(SAFE_DYNC(DataGenIm *,_ptr),*this,true,true);
}

Output GenIm::onotcl()
{
    return new ImOutNotComp(SAFE_DYNC(DataGenIm *,_ptr),*this,false,false);
}

void GenIm::tiff_predictor(INT nb_el,INT nb_ch,INT max_val,bool codage)
{
       SAFE_DYNC(DataGenIm *,_ptr)->tiff_predictor
              (nb_el,nb_ch,max_val,codage);
}


DataGenIm * GenIm::data_im()
{
   return SAFE_DYNC(DataGenIm *,_ptr);
}

const DataGenIm * GenIm::data_im() const
{
   return SAFE_DYNC(const DataGenIm *,_ptr);
}






Elise_Rect GenIm::box() const
{
    DataGenIm *gi = SAFE_DYNC(DataGenIm *,_ptr);

    return Elise_Rect(gi->p0(),gi->p1(),gi->dim());
}



Output GenIm::oper_ass_eg(const OperAssocMixte & op,bool auto_clip)
{

    return new ImRedAssOutNotComp (op,data_im(),*this,auto_clip) ;
     
}
Output GenIm::sum_eg(bool auto_clip)
{
       return oper_ass_eg(OpSum,auto_clip);
}

Output GenIm::histo(bool auto_clip) { return sum_eg(auto_clip); }



Output GenIm::max_eg(bool auto_clip)
{
       return oper_ass_eg(OpMax,auto_clip);
}

Output GenIm::min_eg(bool auto_clip)
{
       return oper_ass_eg(OpMin,auto_clip);
}

Output GenIm::mul_eg(bool auto_clip)
{
       return oper_ass_eg(OpMul,auto_clip);
}


void GenIm::load_file(Elise_File_Im f)
{
    data_im()->load_file(f,*this);
}


GenIm::~GenIm(){}

void * GenIm::data_lin()
{
     return data_im()->data_lin_gen();
}

const INT * GenIm::P1()
{
	return data_im()->p1();
}

bool GenIm::same_dim_and_sz(GenIm I2)
{
    DataGenIm * i1 = data_im();
    DataGenIm * i2 = I2.data_im();

    if (i1->dim() != i2->dim())
       return false;

    for (int d =0; d<i1->dim() ; d++)
        if (
                 (i1->p0()[d] != i2->p0()[d])
              && (i1->p1()[d] != i2->p1()[d])
           )
           return false;

    return true;
}

void GenIm::read_data(ELISE_fp & fp)
{
   data_im()->read_data(fp);
}

void GenIm::write_data(ELISE_fp & fp) const
{
   data_im()->write_data(fp);
}



string type_elToString(GenIm::type_el i_type)
{
	switch (i_type)
	{
	case GenIm::u_int1: return "u_int1";
	case GenIm::int1: return "int1";
	case GenIm::u_int2: return "u_int2";
	case GenIm::int2: return "int2";
	case GenIm::int4: return "int4";
	case GenIm::real4: return "real4";
	case GenIm::real8: return "real8";
	case GenIm::bits1_msbf: return "bits1_msbf";
	case GenIm::bits2_msbf: return "bits2_msbf";
	case GenIm::bits4_msbf: return "bits4_msbf";
	case GenIm::bits1_lsbf: return "bits1_lsbf";
	case GenIm::bits2_lsbf: return "bits2_lsbf";
	case GenIm::bits4_lsbf: return "bits4_lsbf";
	case GenIm::real16: return "real_16";
	case GenIm::int8: return "int8";
	case GenIm::u_int4: return "u_int4";
	case GenIm::u_int8: return "u_int8";
	case GenIm::no_type: return "no_type";
	}
	return "unknown";
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
