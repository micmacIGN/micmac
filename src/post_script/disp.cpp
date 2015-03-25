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
#include <ctime>


      //=======   Name of data and procedures defined  =====

#define IMAT     "IMat"
#define DIM      "DIM"

#define DrRect "DrRect"
#define DrCirc "DrCirc"

#define Ix0  "Ix0"
#define Iy0  "Iy0"
#define Itx  "Itx"
#define ULIM "ULIM"

#define IUC   "IUC"
#define IRLE  "IRLE"
#define ILZW  "ILZW"

#define UC1   "UC1"
#define RLE1  "RLE1"
#define LZW1  "LZW1"


#define MIUC   "MIUC"
#define MIRLE  "MIRLE"
#define MILZW  "MILZW"


#define  STRX   "Strx"
#define  STRY   "Stry"
#define  STRC0  "Strc0"
#define  STRC1  "Strc1"
#define  STRC2  "Strc2"

#define  LSt85  "LSt85"

#define LptsImCste    "LptsImCste"
#define LptsImInd     "LptsImInd"
#define LptsImGray    "LptsImGray"
#define LptsImRGB     "LptsImRGB"

#define  PsPolyline "Polyl"
#define  PsPolyFxy  "PolFxy"

         //======= Name + ElPsPREFIX

#define pIMAT  ElPsPREFIX IMAT
#define pDIM   ElPsPREFIX DIM 

#define pIx0  ElPsPREFIX Ix0
#define pIy0  ElPsPREFIX Iy0
#define pItx  ElPsPREFIX Itx

#define pULIM ElPsPREFIX ULIM


#define  pSTRX   ElPsPREFIX STRX 
#define  pSTRY   ElPsPREFIX STRY 
#define  pSTRC0  ElPsPREFIX STRC0 
#define  pSTRC1  ElPsPREFIX STRC1 
#define  pSTRC2  ElPsPREFIX STRC2 

         //======== DATA ============

#define Def_DIM "<</ImageType 1  /ImageMatrix " pIMAT ">>"
#define Def_IMAT "[1 0 0 1 0 0]"

         //========= PROCEDURES ====================

                //== AUXILAIRY (for long procedures) === 

#define  ASCII85 " currentfile \n /ASCII85Decode filter "
#define  IMA85F  " /DataSource "  ASCII85

               // ===  procedures  body

#define Def_Ix0  "{" pIMAT " 4  3 -1 roll neg put} bind"
#define Def_Iy0  "{" pIMAT " 5  3 -1 roll neg put} bind"
#define Def_Itx  "{" pDIM " /Width 3 -1 roll put} bind"
#define Def_Ity  "{" pDIM " /Height 3 -1 roll put} bind"
#define Def_Inbb "{" pDIM " /BitsPerComponent 3 -1 roll put} bind"


#define Def_IUC    "{" pDIM IMA85F " put " pDIM " image} bind"
#define Def_IRLE   "{" pDIM IMA85F " /RunLengthDecode filter put " pDIM " image} bind"
#define Def_ILZW   "{" pDIM IMA85F " /LZWDecode filter put " pDIM " image} bind"


#define Def_MIUC    "{true " pIMAT ASCII85 "  imagemask} bind"
#define Def_MIRLE   "{true " pIMAT ASCII85 " /RunLengthDecode filter  imagemask} bind"
#define Def_MILZW   "{true " pIMAT ASCII85 " /LZWDecode filter  imagemask} bind"



#define Def_ULIM  "{" pIx0 " " pIy0 " " pItx "} bind"

#define Def_UC1  "{" pULIM " " ElPsPREFIX IUC "} bind"
#define Def_RLE1 "{" pULIM " " ElPsPREFIX IRLE "} bind"
#define Def_LZW1 "{" pULIM " " ElPsPREFIX ILZW "} bind"
#define Def_L "{newpath moveto lineto stroke} bind"
#define Def_DrRect   "{newpath moveto dup 0 rlineto exch 0 exch rlineto\n neg 0 rlineto closepath stroke}"

#define Def_DrCirc "{newpath   0 360 arc stroke}"

#define  Def_LSt85  "{" ASCII85 " exch readstring} bind"


char BUF_STR[20];
#define CHAINE_STR " %d string "

const char *   Def_LptsImCste  
       =
         "{\n"\
         "  0 1  3 -1 roll\n"
         "  {\n"
         "     dup dup \n"
         "     " pSTRX "  exch  get " pSTRY " 3 -1 roll   get  " pSTRC0 " 4 -1 roll get 1 add  1 rectfill\n"
         "  }\n"
         "  for clear\n"
         "}  bind";


const char *   Def_LptsImInd  
       =
         "{\n"\
         "  0 1  3 -1 roll\n"
         "  {\n"
         "     dup dup \n"
         "     " pSTRC0 "  exch get setcolor\n"
         "     " pSTRX "  exch  get " pSTRY " 3 -1 roll   get 1 1 rectfill\n"
         "  }\n"
         "  for clear\n"
         "}  bind";

const char *   Def_LptsImGray  
       =
         "{\n"\
         " 0 1  3  -1 roll\n"
         "  {\n"
         "     dup dup \n"
         "     " pSTRC0 "  exch get 256 div setgray\n"
         "     " pSTRX "  exch  get " pSTRY " 3 -1 roll   get 1 1 rectfill\n"
         "  }\n"
         "  for clear\n"
         "}  bind";

const char *   Def_LptsImRGB  
       =
         "{\n"\
         "  0 1  3 -1 roll\n"
         "  {\n"
         "     dup dup dup dup\n"
         "        " pSTRC0 "  exch get 256 div \n"
         "        " pSTRC1 "  3 -1 roll  get 256 div \n"
         "        " pSTRC2 "  4 -1 roll  get 256 div \n"
         "     setrgbcolor \n"
         "     " pSTRX "  exch  get " pSTRY " 3 -1 roll   get 1 1 rectfill\n"
         "  }\n"
         "  for clear\n"
         "}  bind";

const char * Def_Polyline = "{newpath moveto {rlineto} repeat stroke}";
const char * Def_PolFxy   = "{newpath moveto {dup  3 -1 roll  rlineto} repeat stroke pop}";



/************************************************************************/
/************************************************************************/
/************************************************************************/



/************************************************************************/
/*                                                                      */
/*             PS_Display::def                                          */
/*                                                                      */
/************************************************************************/

Data_Elise_PS_Disp::defV::defV
(
     const char *      name, 
     const char *      proc,
     defV_ul_action    use_action,
     defV_ul_action    load_action
)    :
     _name       (name),
     _act_use    (use_action),
     _act_load   (load_action),
     _init       (false),
     _proc       (proc)
{
}

void Data_Elise_PS_Disp::defV::no_act_init(Data_Elise_PS_Disp *)
{
}


void Data_Elise_PS_Disp::defV::act_use_1LigI(Data_Elise_PS_Disp * psd)
{
    psd->_x0Im.over_prim(psd);
    psd->_y0Im.over_prim(psd);
    psd->_txIm.over_prim(psd);
    psd->_1LigI.load_prim(psd);
}

void Data_Elise_PS_Disp::defV::act_use_F1Ucomp(Data_Elise_PS_Disp * psd)
{
     act_use_1LigI(psd);
     psd->_FUcomp.load_prim(psd);
}

void Data_Elise_PS_Disp::defV::act_use_F1RLE(Data_Elise_PS_Disp * psd)
{
     act_use_1LigI(psd);
     psd->_FRLE.load_prim(psd);
}

void Data_Elise_PS_Disp::defV::act_use_F1LZW(Data_Elise_PS_Disp * psd)
{
     act_use_1LigI(psd);
     psd->_FLZW.load_prim(psd);
}



const Data_Elise_PS_Disp::defV_ul_action  
       Data_Elise_PS_Disp::NO_ACT_INIT 
     = Data_Elise_PS_Disp::defV::no_act_init;

Data_Elise_PS_Disp::defF::defF
(
     const char * name, 
     const char * proc,
     defV_ul_action    use_action,
     defV_ul_action    load_action
)    :
     defV(name,proc,use_action,load_action)
{
}

Data_Elise_PS_Disp::defI::defI
(
     const char * name, 
     const char * proc
)    :
     defV(name,proc),
     _i  (-0x7FFFFFFF)
{
}


Data_Elise_PS_Disp::RdefF::RdefF() :
      _defF (0)
{
}



void Data_Elise_PS_Disp::defV::load_prim(Data_Elise_PS_Disp * psd)
{
    if (! _init)
    {
        _init = true;
        _act_load(psd);
        psd->_fh << "/" << ElPsPREFIX << _name
                 << " " << _proc 
                 << " def \n";
    }
}

void Data_Elise_PS_Disp::defF::put_prim(Data_Elise_PS_Disp * psd)
{
    _act_use(psd);
    load_prim(psd);
    psd->_fd << ElPsPREFIX << _name << "\n";
}


void Data_Elise_PS_Disp::defI::put_prim(Data_Elise_PS_Disp * psd,INT i)
{
    load_prim(psd);
    if (i != _i)
    {
       _i = i;
       psd->_fd  << _i << " " << ElPsPREFIX << _name << "\n";
    }
    _act_use(psd);
}

void Data_Elise_PS_Disp::defI::over_prim(Data_Elise_PS_Disp * psd)
{
    load_prim(psd);
    _i = -0x7fffffff;
}

void Data_Elise_PS_Disp::RdefF::put_prim(Data_Elise_PS_Disp * psd,defF * newdefF)
{
    if (_defF != newdefF)
    {
         newdefF->put_prim(psd);
         _defF = newdefF;
    }
}




/************************************************************************/
/*                                                                      */
/*             PS_Display                                               */
/*                                                                      */
/************************************************************************/


// Je ne sais plus du tout d'ou vient cette valeur. 
// Peut etre d'un vielle mesure ? A affiner eventuellement

const REAL PS_Display::picaPcm = 451.0 / 15.8;
const Pt2dr PS_Display::A4( 21.0,29.7);
const Pt2dr PS_Display::A3( 29.7,42.0);
const Pt2dr PS_Display::A2( 42.0,59.4);
const Pt2dr PS_Display::A1( 59.4,84);
const Pt2dr PS_Display::A0( 84.0,118.8);

PS_Display::PS_Display
(
    const char *             name,
    const char *             title,
    Elise_Set_Of_Palette     sop,
    bool                     auth_lzw,
    Pt2dr                    szp

)  :
   PRC0(new Data_Elise_PS_Disp(name,title,sop,auth_lzw,szp))
{
}


void PS_Display::comment(const char * com)
{
     depsd()->comment(com);
}

/************************************************************************/
/*                                                                      */
/*          Data_Elise_PS_Disp                                          */
/*                                                                      */
/************************************************************************/

void Data_Elise_PS_Disp::comment(const char * com)
{
    if ((com == 0) || (*com == 0))
    {
       _fd << "%\n";
       return;
    }

    _fd << "%";
    for (;*com;com++)
    {
        _fd << *com;
        if (*com == '\n')
           _fd << "%";
    }
    _fd << "\n";
}

            //=======================================

void Data_Elise_PS_Disp::Lpts_put_prim(bool cste)
{
     if (cste)
        _LptsImCste.put_prim(this);
     else
     {
         switch (_act_ps_pal->cdev())
         {
             case Elise_PS_Palette::indexed :
                  _LptsImInd.put_prim(this);
             break;

             case Elise_PS_Palette::gray :
                  _LptsImGray.put_prim(this);
             break;

             case Elise_PS_Palette::rgb :
                  _LptsImRGB.put_prim(this);
             break;

             default :
                El_Internal.ElAssert(false,EEM0<<"PS_Disp::Lpts_put_prim");
         }
     }
}




void Data_Elise_PS_Disp::_inst_set_line_witdh(REAL lw)
{
     _fd <<  lw/10.0 << " setlinewidth\n";
}



Elise_PS_Palette * Data_Elise_PS_Disp::get_ps_pal(Data_Elise_Palette * dep)
{
    for (int i= 0; i<_nbpal; i++)
        if( _teps[i]->_pal == dep)
          return _teps[i];

    Tjs_El_User.ElAssert
    (
        false,
        EEM0 << "Invalide palette in PS file \n"
    );

    return 0;
}

void Data_Elise_PS_Disp::use_conv_colors
     (
           Data_Elise_Palette * dep,
           U_INT1 ** in,
           INT ** out,
           INT dim,
           INT nb
     )
{
    get_ps_pal(dep)->use_colors(in,out,dim,nb);
}
void Data_Elise_PS_Disp::set_active_palette(Elise_Palette p,bool image)
{
    Data_Elise_Palette * dep = p.dep();

    if (_active_pal == dep)
    {
       if (image)
           _act_ps_pal->_image_used = true;
       return;
    }
    _active_pal = dep;

    _act_ps_pal = get_ps_pal(dep);
    _act_ps_pal->load(_fd,image);
}

void Data_Elise_PS_Disp::set_cur_color(const INT * c)
{
     _act_ps_pal->set_cur_color(_fd,c);
}




Data_Elise_PS_Disp::Data_Elise_PS_Disp
       (
               const char *          name,
               const char *          title,
               Elise_Set_Of_Palette  sop,
               bool                  auth_lzw,
               Pt2dr                 sz_page
       )    :
       Data_Elise_Gra_Disp  ()               ,
       _name                (dup(name)),
       _fp                  (name,ios::out)  ,
       _name_data           (cat(name,".psd")),
       _fd                  (_name_data,ios::out),
       _name_header         (cat(name,".psh")),
       _fh                  (_name_header,ios::out),
       _sop                 (sop)            ,
       _offs_bbox           (-1)             ,
       _p0_box              (0x7777,0x7777),
       _p1_box              (0x7777,0x7777),
       _nb_win              (0),
       _nbpal               (sop.lp().card())     ,
       _teps                (NEW_VECTEUR(0,_nbpal,Elise_PS_Palette *)),
       _sz_page             (sz_page),
       _num_last_act_win    (-1),
       _active_pal          (0),
       _act_ps_pal          (0),
       _use_lzw             (auth_lzw),
       _use_pckb            (true),

       _FUcomp              (IUC,Def_IUC),
       _FRLE                (IRLE,Def_IRLE),
       _FLZW                (ILZW,Def_ILZW),
       _1LigI               (ULIM,Def_ULIM,defV::act_use_1LigI),
       _F1Ucomp             (UC1,Def_UC1,defV::act_use_F1Ucomp),
       _F1RLE               (RLE1,Def_RLE1,defV::act_use_F1RLE),
       _F1LZW               (LZW1,Def_LZW1,defV::act_use_F1LZW),

       _MUcomp             (MIUC ,Def_MIUC),
       _MRLE               (MIRLE,Def_MIRLE),
       _MLZW               (MILZW,Def_MILZW),


       _line                ("L",Def_L),
       _dr_circ             (DrCirc,Def_DrCirc),
       _dr_rect             (DrRect,Def_DrRect),
       _dr_poly             (PsPolyline,Def_Polyline),
       _dr_polyFxy          (PsPolyFxy,Def_PolFxy),

       _DicIm               (DIM,Def_DIM),
       _MatIm               (IMAT,Def_IMAT),

       _StrX                (STRX,BUF_STR),
       _StrY                (STRY,BUF_STR),
       _StrC0               (STRC0,BUF_STR),
       _StrC1               (STRC1,BUF_STR),
       _StrC2               (STRC2,BUF_STR),
       _LStr85              (LSt85,Def_LSt85),

       _LptsImCste          (LptsImCste,Def_LptsImCste),
       _LptsImInd           (LptsImInd,Def_LptsImInd),
       _LptsImGray          (LptsImGray,Def_LptsImGray),
       _LptsImRGB           (LptsImRGB,Def_LptsImRGB),

       _x0Im                (Ix0,Def_Ix0),
       _y0Im                (Iy0,Def_Iy0),
       _txIm                (Itx,Def_Itx),
       _tyIm                ("Ity",Def_Ity),
       _nbbIm               ("Inbb",Def_Inbb),


       _lclip               (true),
       _lgeo_clip           (false)

                                   
{
/*
     if ( (!_use_lzw) || (! _use_pckb))
        cout << "WARNSS , compression inhibee \n";
*/
	int i;

       sprintf(BUF_STR,CHAINE_STR,max_str);
       _StrC[0] = &_StrC0;
       _StrC[1] = &_StrC1;
       _StrC[2] = &_StrC2;


      _fp << "%!PS-Adobe-2.0\n";
      _fp << "%%BoundingBox: ";
      _offs_bbox = (INT)_fp.tellp();
      _fp << "                                                                        \n";
      _fp << "%%Creator : ELISE.0.0\n";
      _fp << "%%Title :" << title << "\n";

       {
           time_t t= time(0);
           tm * td = localtime(&t);

           _fp << "%%CreationDate: "
               << td->tm_mday << "/" << td->tm_mon << "/"  << td->tm_year << " "
               << td->tm_hour << "h" << td->tm_min << "m"  << td->tm_sec << "s\n";

       }

       _fp << "%%EndComments\n";
       _fp << "%%Pages: 1\n";

       _fp << "/EliseSaveAllObject save def\n";

       //=============================

      _fh << "\n\n% Definition of curent primitives : \n\n";

       for ( i=0; i<_nbpal; i++)
           _teps[i] = 0;

       L_El_Palette   lp = sop.lp();
       for ( i=0; i<_nbpal; i++)
       {
           char buf[200];
           sprintf(buf,"%sPal%d",ElPsPREFIX,i);
           _teps[i] = lp.car().ps_comp(buf);
           lp = lp.cdr();
       }
       // because load of palette fixe decode of DIM, even when no images are used

       //=============================

       _fd << "\n\n%Drawings orders :\n\n";
       _fd << "gsave\n";

       //=============================

       _MatIm.load_prim(this);
       _DicIm.load_prim(this);
}

      

void Data_Elise_PS_Disp::add_file(ofstream &f,const char * name)
{
    INT nb = (INT)f.tellp();
    f.close();
    ifstream   src(name,ios::in);

    for(int i=0 ; i< nb ; i++)
         _fp.put((char)src.get());
}

Data_Elise_PS_Disp::~Data_Elise_PS_Disp()
{
	int i;

   _fh << "\n\n% Definition of PS-palette \n";
   for ( i=0; i<_nbpal; i++)
       if (_teps[i]->_used)
       {
             _teps[i]->_pal->ps_end(_teps[i],_fh);
              _fh << "/";
              _teps[i]->load(_fh,false); 
              _fh << "{";
              _teps[i]->load_def(_fh);
              if (_teps[i]->_image_used)
              {
                  _fh << " " << pDIM << " /Decode ";
                  _teps[i]->im_decode(_fh);
                  _fh << " put"; 
              }
              _fh <<"} def \n";
       }
   add_file(_fh,_name_header);

   _fp << "%%EndProlog \n";
   _fp << "%%Page: 1 1\n";

   add_file(_fd,_name_data);


   _fd << "grestore\n";
   _fp << "showpage\n";

   _fp << "EliseSaveAllObject restore\n";
   _fp << "grestoreall\n";

   _fp << "%%Trailer\n";

   _fp.close();
   _fp.open(_name,ios::in|ios::out);
   _fp.seekp(_offs_bbox,ios::beg);

   _fp 
        << round_ni(_p0_box.x) << " "
        << round_ni(_p0_box.y) << " "
        << round_ni(_p1_box.x) << " "
        << round_ni(_p1_box.y) ;

    for ( i=_nbpal-1; i>=0 ; i--)
        if (_teps[i])
           _teps[i]->~Elise_PS_Palette();
    
    DELETE_VECTOR(_teps,0);
    DELETE_VECTOR(_name_data,0);
    DELETE_VECTOR(_name_header,0);
    DELETE_VECTOR(_name,0);
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
