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


/***********************************************************/
/***********************************************************/
/***                                                     ***/
/***           GENERAL CLASSES                           ***/
/***                                                     ***/
/***********************************************************/
/***********************************************************/


        /*========================================
        ||       Data_Arg_Tiff                   ||
        =========================================*/

class Data_Arg_Tiff :  public RC_Object
{
   friend class D_Tiff_ifd_Arg_opt;
   private :
      virtual void update_tiff(D_Tiff_ifd_Arg_opt *) = 0;
};

        /*========================================
        ||         Arg_Tiff                      ||
        =========================================*/

Data_Arg_Tiff * Arg_Tiff::dat() const
{
      {return SAFE_DYNC(class Data_Arg_Tiff *,_ptr);}
};

Arg_Tiff::Arg_Tiff(Data_Arg_Tiff * dat) :
    PRC0(dat)
{
}

        /*========================================
        ||       D_Tiff_ifd_Arg_opt              ||
        =========================================*/

int Tiff_Im::mDefTileFile = 1 << 18;
void Tiff_Im::SetDefTileFile(int aSz)
{
   mDefTileFile = aSz;
}
int Tiff_Im::DefTileFile()
{
   return  mDefTileFile;
}

D_Tiff_ifd_Arg_opt::D_Tiff_ifd_Arg_opt() :
    mExifTiff_Date(cElDate::NoDate)
{
     _predictor     = -1;
     _sz_tile       = Pt2di(-1,-1);
     _row_per_strip = -1;
     _no_strip      = 0;
     _plan_conf     = -1;
     _init_min_maxs = false;
     _res_unit      = -1;
     _orientation   = 1;
     mSzFileTile = Pt2di(Tiff_Im::DefTileFile(),Tiff_Im::DefTileFile());

     mExifTiff_FocalLength = -1;
     mExifTiff_FocalEqui35Length = -1;
     mExifTiff_ShutterSpeed = -1;
     mExifTiff_Aperture = -1;
     mExifTiff_IsoSpeed = -1;
}

void D_Tiff_ifd_Arg_opt::modif(L_Arg_Opt_Tiff l)
{
     for(;! (l.empty()); l = l.cdr())
     {
         l.car().dat()->update_tiff(this);
     }
}


/***********************************************************/
/***********************************************************/
/***                                                     ***/
/***           Pt2di classes                             ***/
/***                                                     ***/
/***********************************************************/
/***********************************************************/

/*
    To economize the number of classes, we give integrate
   the classes with integer values in this classes (=
   x code the value and y is not used.
*/



class Data_Atiff_Pt2di : public Data_Arg_Tiff
{
     // friend class PlModePl;

     public :

        typedef enum  mode
        {
             mode_predictor,
             mode_sz_tile,
             mode_row_per_str,
             mode_no_strip,
             mode_plan_conf,
             mode_min_max_sample,
             mode_resol,
             mode_orientation,
	     mode_sz_file_tile,
             mode_ExifTiff_FocalLength,
             mode_ExifTiff_FocalEqui35Length,
             mode_ExifTiff_ShutterSpeed,
             mode_ExifTiff_Aperture,
             mode_ExifTiff_IsoSpeed,
             mode_ExifTiff_Date,
             mode_ExifTiff_Camera
        } mode_modif;

     public :

        Data_Atiff_Pt2di(Pt2dr ptr,INT i,mode_modif mode) : 
               _ptr (ptr), 
               _i   (i), 
               _d   (cElDate::NoDate),
               _mode(mode) 
        {}

        Data_Atiff_Pt2di(Pt2di pt,mode_modif mode) : 
               _pt (pt), 
               _d   (cElDate::NoDate),
               _mode(mode) 
        {}

// Pour  INT et double, on stocke les 2, par compat avec apparition tardive de double
        Data_Atiff_Pt2di(INT v,mode_modif mode) : 
                _pt  (v,0), 
                _ptr  (v,0), 
               _d   (cElDate::NoDate),
               _mode (mode) 
        {}

        Data_Atiff_Pt2di(double v,mode_modif mode) : 
                _pt   (round_ni(v),0), 
                _ptr  (v,0), 
               _d   (cElDate::NoDate),
               _mode (mode) 
        {}







        Data_Atiff_Pt2di(mode_modif mode) : 
               _pt   (0,0), 
               _d   (cElDate::NoDate),
               _mode (mode) 
        {}

        Data_Atiff_Pt2di(const cElDate & aDate ,mode_modif mode) : 
               _d    (aDate),
               _mode (mode) 
        {}
        Data_Atiff_Pt2di(const std::string &  s,mode_modif mode) : 
               _d    (cElDate::NoDate),
               _s    (s),
               _mode (mode) 
        {}






     private :
        virtual void update_tiff(D_Tiff_ifd_Arg_opt * dtiao)
        {

           switch (_mode)
           {
              case mode_predictor : 
                   dtiao->_predictor = _pt.x; 
              break;

              case mode_sz_tile:
// std::cout << "MODIF TO SZ TILEE\n";
                   if (_pt.x <0)
                   {
                        dtiao->_sz_tile = - _pt;
                   }
                   else
                   {
                       dtiao->_sz_tile = Pt2di
                                    (
                                         round_up(_pt.x,16),
                                         round_up(_pt.y,16)
                                    );
                   }
              break;

              case mode_sz_file_tile :
	           if (_pt.x<0)
		      dtiao->mSzFileTile = _pt;
	           else
	           {
	             if (dtiao->mSzFileTile.x >0)
                        dtiao->mSzFileTile = Pt2di
                                      (
                                         round_up(_pt.x,16),
                                         round_up(_pt.y,16)
                                      );
                   }
              break;

              case mode_row_per_str:
                   dtiao->_row_per_strip = _pt.x; 
              break;

              case mode_no_strip :
                   dtiao->_no_strip = 1; 
              break;

              case mode_plan_conf :
                   dtiao->_plan_conf = _pt.x; 
              break;

              case mode_min_max_sample :
                   dtiao->_init_min_maxs = true;
                   dtiao->_mins          = _pt.x;
                   dtiao->_maxs          = _pt.y;
              break;

              case mode_resol :
                   dtiao->_res_unit = _i;
                   dtiao->_resol    = _ptr;
              break;

              case mode_orientation :
                   dtiao->_orientation = _pt.x;
              break;

              case mode_ExifTiff_FocalLength :
                   dtiao->mExifTiff_FocalLength = _ptr.x;
              break;

              case mode_ExifTiff_FocalEqui35Length :
                   dtiao->mExifTiff_FocalEqui35Length = _ptr.x;
              break;

              case mode_ExifTiff_ShutterSpeed :
                   dtiao->mExifTiff_ShutterSpeed = _ptr.x;
              break;


              case mode_ExifTiff_Aperture :
                   dtiao->mExifTiff_Aperture = _ptr.x;
              break;

              case mode_ExifTiff_IsoSpeed :
                   dtiao->mExifTiff_IsoSpeed = _ptr.x;
              break;

              case mode_ExifTiff_Date :
                   dtiao->mExifTiff_Date = _d;
              break;

              case mode_ExifTiff_Camera :
                   dtiao->mExifTiff_Camera = _s;
              break;
           }
        }

        Pt2di           _pt;
        Pt2dr           _ptr;
        INT             _i;
        cElDate         _d;
        std::string     _s;
        mode            _mode;
};



Tiff_Im::APred::APred(Tiff_Im::PREDICTOR pred) :
    Arg_Tiff
    ( new Data_Atiff_Pt2di
      (pred,Data_Atiff_Pt2di::mode_predictor)
    )
{
}


Tiff_Im::ATiles::ATiles(Pt2di szt) :
    Arg_Tiff
    ( new Data_Atiff_Pt2di
      (szt,Data_Atiff_Pt2di::mode_sz_tile)
    )
{
}

Tiff_Im::AFileTiling::AFileTiling(Pt2di szt) :
    Arg_Tiff
    ( new Data_Atiff_Pt2di
      (szt,Data_Atiff_Pt2di::mode_sz_file_tile)
    )
{
}



Tiff_Im::AStrip::AStrip(INT rps) :
    Arg_Tiff
    ( new Data_Atiff_Pt2di
      (rps,Data_Atiff_Pt2di::mode_row_per_str)
    )
{
}

Tiff_Im::AOrientation::AOrientation(INT orient) :
    Arg_Tiff
    ( new Data_Atiff_Pt2di
      (orient,Data_Atiff_Pt2di::mode_orientation)
    )
{
}





Tiff_Im::ANoStrip::ANoStrip() :
    Arg_Tiff
    ( new Data_Atiff_Pt2di
      (Data_Atiff_Pt2di::mode_no_strip)
    )
{
}


Tiff_Im::APlanConf::APlanConf(Tiff_Im::PLANAR_CONFIG pl_conf) :
    Arg_Tiff
    ( new Data_Atiff_Pt2di
      (pl_conf,Data_Atiff_Pt2di::mode_plan_conf)
    )
{
}



Tiff_Im::AMinMax::AMinMax(U_INT2 vmin,U_INT2 vmax) :
    Arg_Tiff
    ( new Data_Atiff_Pt2di
      (Pt2di(vmin,vmax),Data_Atiff_Pt2di::mode_min_max_sample)
    )
{
}


Tiff_Im::AResol::AResol(REAL res,RESOLUTION_UNIT unit) :
    Arg_Tiff
    ( new Data_Atiff_Pt2di
      (Pt2dr(res,res),unit,Data_Atiff_Pt2di::mode_resol)
    )
{
}

Tiff_Im::AExifTiff_FocalLength::AExifTiff_FocalLength(double aF) :
   Arg_Tiff
   (
       new Data_Atiff_Pt2di(aF,Data_Atiff_Pt2di::mode_ExifTiff_FocalLength)
   )
{
}


Tiff_Im::AExifTiff_FocalEqui35Length::AExifTiff_FocalEqui35Length(double aF) :
   Arg_Tiff
   (
       new Data_Atiff_Pt2di(aF,Data_Atiff_Pt2di::mode_ExifTiff_FocalEqui35Length)
   )
{
}





Tiff_Im::AExifTiff_ShutterSpeed::AExifTiff_ShutterSpeed(double aF) :
   Arg_Tiff
   (
       new Data_Atiff_Pt2di(aF,Data_Atiff_Pt2di::mode_ExifTiff_ShutterSpeed)
   )
{
}






Tiff_Im::AExifTiff_Aperture::AExifTiff_Aperture(double aA) :
   Arg_Tiff
   (
       new Data_Atiff_Pt2di(aA,Data_Atiff_Pt2di::mode_ExifTiff_Aperture)
   )
{
}

Tiff_Im::AExifTiff_IsoSpeed::AExifTiff_IsoSpeed(double aI) :
   Arg_Tiff
   (
       new Data_Atiff_Pt2di(aI,Data_Atiff_Pt2di::mode_ExifTiff_IsoSpeed)
   )
{
}

Tiff_Im::AExifTiff_Date::AExifTiff_Date(const cElDate &  aD) :
   Arg_Tiff
   (
       new Data_Atiff_Pt2di(aD,Data_Atiff_Pt2di::mode_ExifTiff_Date)
   )
{
}

Tiff_Im::AExifTiff_Camera::AExifTiff_Camera(const std::string &  aC) :
   Arg_Tiff
   (
       new Data_Atiff_Pt2di(aC,Data_Atiff_Pt2di::mode_ExifTiff_Camera)
   )
{
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
